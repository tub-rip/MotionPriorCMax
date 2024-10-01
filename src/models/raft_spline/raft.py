from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn

from .parts.extractor import BasicEncoder
from .parts.update import BasicUpdateBlock
from .curves import CurveFactory, CurveBase
from .utils import coords_grid
from .corr import CorrComputation, CorrBlockParallelMultiTarget

class RAFTSpline(nn.Module):
    def __init__(self, model_params: Dict[str, Any], basis_network=None):
        super().__init__()
        nbins_context = model_params['num_bins']['context']
        nbins_correlation = model_params['num_bins']['correlation']
        self.bezier_degree = model_params['bezier_degree']
        self.detach_bezier = model_params['detach_bezier']
        self.curve_type = model_params['curve_type']
        self.basis_network = basis_network

        print(f'Detach Bezier curves: {self.detach_bezier}')

        assert nbins_correlation > 0 and nbins_context > 0
        assert self.bezier_degree >= 1
        self.nbins_context = nbins_context
        self.nbins_corr = nbins_correlation

        print('RAFT-Spline config:')
        print(f'Num bins context: {nbins_context}')
        print(f'Num bins correlation: {nbins_correlation}')

        corr_params = model_params['correlation']
        self.corr_use_cosine_sim = corr_params['use_cosine_sim']

        ev_corr_params = corr_params['ev']
        self.ev_corr_target_indices = ev_corr_params['target_indices']
        self.ev_corr_levels = ev_corr_params['levels']
        self.ev_corr_radius = 4

        self.img_corr_params = None
        if model_params['use_boundary_images']:
            print('Using images')
            self.img_corr_params = corr_params['img']
            assert 'levels' in self.img_corr_params
            assert 'radius' in self.img_corr_params

        self.hidden_dim = hdim = model_params['hidden']['dim']
        self.context_dim = cdim = model_params['context']['dim']
        cnorm = model_params['context']['norm']
        feature_dim = model_params['feature']['dim']
        fnorm = model_params['feature']['norm']

        # feature network, context network, and update block
        context_dim = 0
        self.fnet_img = None
        if self.img_corr_params is not None:
            self.fnet_img = BasicEncoder(input_dim=3, output_dim=feature_dim, norm_fn=fnorm)
            context_dim += 3
        self.fnet_ev = None
        if model_params['use_events']:
            print('Using events')
            assert 0 not in self.ev_corr_target_indices
            assert len(self.ev_corr_target_indices) > 0
            assert max(self.ev_corr_target_indices) < self.nbins_context
            assert len(self.ev_corr_target_indices)  == len(self.ev_corr_levels)
            self.fnet_ev = BasicEncoder(input_dim=nbins_correlation, output_dim=feature_dim, norm_fn=fnorm)
            context_dim += nbins_context
        assert self.fnet_ev is not None or self.fnet_img is not None
        self.cnet = BasicEncoder(input_dim=context_dim, output_dim=hdim + cdim, norm_fn=cnorm)

        self.update_block = BasicUpdateBlock(model_params, hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, input_):
        N, _, H, W = input_.shape
        # batch, 2, ht, wd
        downsample_factor = 8
        coords0 = coords_grid(N, H//downsample_factor, W//downsample_factor, device=input_.device)
        bezier = CurveFactory(self.basis_network).create_from_voxel_grid(self.curve_type,
                                                                         input_,
                                                                         downsample_factor=downsample_factor,
                                                                         bezier_degree=self.bezier_degree)
        return coords0, bezier

    def gen_voxel_grids(self, input_: torch.Tensor):
        # input_: N, nbins_context + nbins_corr - 1 , H, W
        assert self.nbins_context + self.nbins_corr - 1 == input_.shape[-3]
        corr_grids = list()
        # We need to add the reference index (which is 0).
        indices_with_reference = [0]
        indices_with_reference.extend(self.ev_corr_target_indices)
        for idx in indices_with_reference:
            slice_ = input_[:, idx:idx+self.nbins_corr, ...]
            corr_grids.append(slice_)
        context_grid = input_[:, -self.nbins_context:, ...]
        return corr_grids, context_grid

    def forward(self,
                voxel_grid: Optional[torch.Tensor]=None,
                images: Optional[List[torch.Tensor]]=None,
                iters: int=12,
                flow_init: Optional[CurveBase]=None,
                test_mode: bool=False):
        assert voxel_grid is not None or images is not None
        assert iters > 0
        assert iters == 12

        hdim = self.hidden_dim
        cdim = self.context_dim
        current_device = voxel_grid.device if voxel_grid is not None else images[0].device

        corr_computation_events = None
        context_input = None

        if self.fnet_ev is not None:
            assert voxel_grid is not None
            voxel_grid = voxel_grid.contiguous()
            corr_grids, context_input = self.gen_voxel_grids(voxel_grid)

            fmaps_ev = self.fnet_ev(corr_grids)
            fmaps_ev = [x.float() for x in fmaps_ev]
            fmap1_ev = fmaps_ev[0]
            fmap2_ev = torch.stack(fmaps_ev[1:], dim=0)
            corr_computation_events = CorrComputation(fmap1_ev, fmap2_ev, num_levels_per_target=self.ev_corr_levels)

        corr_computation_frames = None

        if self.fnet_img is not None:
            assert self.img_corr_params is not None
            assert len(images) == 2
            # images[0]: at reference time
            # images[1]: at target time
            images = [2 * (x.float().contiguous() / 255) - 1 for x in images]
            fmaps_img = self.fnet_img(images)
            corr_computation_frames = CorrComputation(fmaps_img[0], fmaps_img[1], num_levels_per_target=self.img_corr_params['levels'])
            if context_input is not None:
                context_input = torch.cat((context_input, images[0]), dim=-3)
            else:
                context_input = images[0]
        assert context_input is not None

        cnet = self.cnet(context_input)
        net, inp = torch.split(cnet, [hdim, cdim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        # (batch, 2, ht, wd), ...
        coords0, bezier = self.initialize_flow(context_input)

        if flow_init is not None:
            bezier.delta_update_params(flow_init.get_params())

        bezier_up_predictions = []
        dt = 1/(self.nbins_context - 1)

        corr_block = CorrBlockParallelMultiTarget(
            corr_computation_events=corr_computation_events,
            corr_computation_frames=corr_computation_frames)

        for itr in range(iters):
            if self.detach_bezier:
                bezier.detach_()

            lookup_timestamps = list()
            if corr_computation_events is not None:
                for tindex in self.ev_corr_target_indices:
                    # 0 < time <= 1
                    time = dt*tindex
                    lookup_timestamps.append(time)
            if corr_computation_frames is not None:
                lookup_timestamps.append(1)

            flows = bezier.get_flow_from_reference(time=lookup_timestamps)
            coords1 = coords0 + flows
            corr_total = corr_block(coords1)
            bezier_params = bezier.get_params()
            net, up_mask, delta_bezier = self.update_block(net, inp, corr_total, bezier_params)

            # B(k+1) = B(k) + \Delta(B)
            bezier.delta_update_params(delta_bezier=delta_bezier)

            if not test_mode or itr == iters - 1:
                bezier_up = bezier.create_upsampled(up_mask)
                bezier_up_predictions.append(bezier_up)

        if test_mode:
            return bezier, bezier_up

        return bezier_up_predictions