import torch
from pykeops.torch import LazyTensor

from . import TrajectoryLossBase
from .. import utils

EPS = 1e-9

class FocusLoss(TrajectoryLossBase):
    """
    Implements the Focus Loss of MotionPriorCM (https://arxiv.org/pdf/2407.10802).

    Args:
        image_shape (tuple): Shape of the image as (height, width).
        num_tref (int): Number of reference time points. If 1, uses a random reference time.
        num_bins (int): Number of voxel grid channels.
        num_knn (int): Number of nearest neighbors used in interpolation.
        smooth_weight (float): Weight for smoothness loss.
        lut_superpixel_size (int): Determines size of the flow look-up-table.
        focus_loss_norm (str): Type of norm used in the focus loss ('l1' or 'l2').
        dist_norm (str): Distance norm type ('l1' or 'l2').
        scale_iwe_by_dt (bool): If true, scales the IWE by time differences.
        mask_image_border (bool): If true, applies masking to image borders.
        polarity_aware_batching (bool): If true, calculates positive/negative IWE separately.
        interpolation_scheme (str): Scheme for interpolation ('mean' or 'iwd').
        smooth_type (str): Type of smoothness ('on_flow_to_tref' or 'on_flow_to_next').
    """
    def __init__(self, image_shape, num_tref, num_bins, num_knn, smooth_weight,
                 lut_superpixel_size, focus_loss_norm, dist_norm,
                 scale_iwe_by_dt, mask_image_border, polarity_aware_batching,
                 interpolation_scheme, smooth_type, **kwargs):
        super().__init__()
        self.image_shape = image_shape
        self.num_tref = num_tref
        self.num_bins = num_bins
        self.num_knn = num_knn
        self.smooth_weight = smooth_weight
        self.lut_superpixel_size = lut_superpixel_size
        self.focus_loss_norm = focus_loss_norm
        self.dist_norm = dist_norm
        self.scale_iwe_by_dt = scale_iwe_by_dt
        self.mask_image_border = mask_image_border
        self.polarity_aware_batching = polarity_aware_batching
        self.interpolation_scheme = interpolation_scheme
        self.smooth_type = smooth_type
        self.is_needing_offsets = True
        self.imager = utils.EventImageConverter(self.image_shape)

        assert not scale_iwe_by_dt or num_tref == 1
        assert not polarity_aware_batching or num_tref == 1
        assert not smooth_type == 'on_flow_to_next' or num_tref == 1

    def get_reconstruction_times(self, device):
        if self.num_tref > 1:
            t_ref = torch.linspace(0, 1, self.num_tref, device=device)
        elif self.num_tref == 1:
            t_ref = torch.rand(1, device=device)  # Random reference time            
        else:
            raise ValueError("Invalid value for num_tref. Must be >= 1.")

        t_bins = torch.linspace(0, 1, self.num_bins+1, device=device)
        t_mid = (t_bins[:-1] + t_bins[1:]) / 2  # Mid-points between bin edges

        return torch.concat((t_ref, t_mid), dim=0)

    def calc(self, trajectories, times, batch):
        """
        Computes the focus and smoothness losses based on trajectories and events.

        Args:
            trajectories (torch.Tensor): Predicted trajectories.
            times (torch.Tensor): Time reference points.
            batch (dict): Dictionary containing the batch data (events, etc.).

        Returns:
            tuple: Loss, log metadata, and miscellaneous metadata.
        """
        events = batch['events']
        num_pos_events = batch['num_pos_events'] if 'num_pos_events' in batch else -1
        assert not self.polarity_aware_batching or num_pos_events > -1

        t_ref = times[:self.num_tref]
        traj_at_tref = trajectories[:, :self.num_tref]
        traj_at_tmid = trajectories[:, self.num_tref:]
        
        flow_lut, flow_to_next = self.interpolate_flow(traj_at_tref, traj_at_tmid)
        warped = self.warp_events(events, flow_lut)
        iwes = self.make_iwes(warped, t_ref, num_pos_events)

        focus_loss = utils.calculate_focus_loss(iwes, loss_type='gradient_magnitude',
                                                norm=self.focus_loss_norm)
        smooth_loss = self.calculate_smooth_loss(flow_lut, flow_to_next)

        loss = focus_loss + smooth_loss

        h, w = self.image_shape

        b, n_tref, _, _ = warped.shape
        if self.polarity_aware_batching:
            iwes = iwes.reshape(b, n_tref, 2, h, w)
        else:
            iwes = iwes.reshape(b, n_tref, h, w)

        log_metadata = {
            'focus_loss': focus_loss.detach(),
            'smoothness_loss': smooth_loss.detach(),
        }

        misc_metadata = {
            'iwes': iwes.detach()
        }

        return loss, log_metadata, misc_metadata

    def interpolate_flow(self, traj_at_tref, traj_at_tmid):
        height, width = self.image_shape
        mid_point_offset = float(self.lut_superpixel_size) / 2 - 0.5
        y = torch.arange(0, height, self.lut_superpixel_size, dtype=torch.float32,
                         device=traj_at_tref.device)
        y += mid_point_offset
        x = torch.arange(0, width, self.lut_superpixel_size, dtype=torch.float32,
                         device=traj_at_tref.device)
        x += mid_point_offset
        height_q, width_q = len(y), len(x)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        grid_points = torch.stack((grid_y, grid_x), dim=-1).reshape(-1 , 2)  # [h_q*w_q, 2]

        # Compute distances between lut coordinates and trajectories
        x_i = LazyTensor(grid_points[None].contiguous())
        q_j = LazyTensor(traj_at_tmid[..., None, :].contiguous())

        if self.dist_norm == 'l2':
            dist = ((x_i - q_j) ** 2).sum(-1)
        elif self.dist_norm == 'l1':
            dist = (x_i - q_j).abs().sum(-1)

        ind_k = dist.argKmin(self.num_knn, dim=2).squeeze(3)
        
        # Compute displacement of each trajectory to reference times 
        traj_at_tref = traj_at_tref.permute(0, 2, 1, 3)
        flow_to_tref = traj_at_tref[:, None] - traj_at_tmid[..., None, :]
        b, n_t, n, n_tref, _ = flow_to_tref.shape
        q = dist.shape[3]

        if self.num_knn == 1:
            ind_k_expanded = ind_k[..., None, None].expand(b, n_t, q, n_tref, 2)
            flow_to_tref = flow_to_tref.gather(2, ind_k_expanded)
        else:
            ind_k_expanded = ind_k[..., None, None]
            ind_k_expanded = ind_k_expanded.expand(b, n_t, q, self.num_knn, n_tref, 2)
            flow_to_tref = flow_to_tref[..., None, :, :]
            flow_to_tref = flow_to_tref.expand(-1, -1, -1, self.num_knn, -1, -1)
            flow_to_tref = flow_to_tref.gather(2, ind_k_expanded)

            if self.interpolation_scheme == 'mean':
                flow_to_tref = torch.mean(flow_to_tref, 3)
            elif self.interpolation_scheme == 'iwd':
                with torch.no_grad():
                    dist_k = dist.Kmin(K=self.num_knn, axis=2)
                    dist_weights = 1 / (dist_k + EPS)
                    dist_weights = dist_weights / torch.sum(dist_weights, dim=3, keepdim=True)
                    dist_weights = dist_weights[..., None, None]
                flow_to_tref = torch.sum(dist_weights * flow_to_tref, dim=3)
            else:
                raise ValueError

        b, n_bins, q, n_tref, d = flow_to_tref.shape
        flow_lut = flow_to_tref.reshape(b, n_bins, height_q, width_q, n_tref, d)

        if self.smooth_weight > 0 and self.smooth_type == 'on_flow_to_next':
            flow_to_next = traj_at_tmid[:, 1:] - traj_at_tmid[:, :-1]
            flow_to_next = flow_to_next[..., None, None, :]
            flow_to_next = flow_to_next.expand(-1, -1, -1, self.num_knn, -1, -1)
            flow_to_next = flow_to_next.gather(2, ind_k_expanded[:, :-1])
            flow_to_next = torch.mean(flow_to_next, dim=3)
            flow_to_next = flow_to_next.reshape(b, n_bins-1, height_q, width_q, n_tref, d)
        else:
            flow_to_next = None

        return flow_lut, flow_to_next

    def warp_events(self, events, flow_lut):
        b, m, _ = events.shape
        ib = torch.arange(b).view(-1, 1).expand(b, m)
        it = events[..., 4].to(int)
        iy = (events[..., 0] // self.lut_superpixel_size).to(int)
        ix = (events[..., 1] // self.lut_superpixel_size).to(int)
        differences = flow_lut[ib, it, iy, ix]
        differences = differences.permute(0, 2, 1, 3)

        warped = differences + events[:, None, :, :2]  
        tp = events[:, None, :, 2:]
        tp = tp.expand(-1, self.num_tref, -1, -1)
        warped = torch.cat((warped, tp), dim=3)
        return warped

    def make_iwes(self, warped, t_ref, num_pos_events):
        b, n_tref, m, d = warped.shape
        warped = warped.reshape(-1, m, d)

        with torch.no_grad():
            weights = warped[..., 5] # padding masks

            if self.scale_iwe_by_dt:
                dt = torch.clamp(torch.abs(warped[..., 2] - t_ref), 0, 1)
                weights = (1 - dt) * weights

            if self.mask_image_border:
                border_mask = torch.ones_like(weights)
                border_mask[warped[..., 0] > self.image_shape[0]] = 0
                border_mask[warped[..., 1] > self.image_shape[1]] = 0
                border_mask[warped[..., 0] < 0] = 0
                border_mask[warped[..., 1] < 0] = 0    
                weights = border_mask * weights

        if self.polarity_aware_batching:
            pos_warped = warped[:, :num_pos_events]
            pos_weights = weights[:, :num_pos_events]
            pos_iwes = self.imager.create_iwe(pos_warped, method='bilinear_vote', sigma=1,
                                              weight=pos_weights)

            neg_warped = warped[:, num_pos_events:]
            neg_weights = weights[:, num_pos_events:]
            neg_iwes = self.imager.create_iwe(neg_warped, method='bilinear_vote', sigma=1,
                                              weight=neg_weights)

            return torch.stack((pos_iwes, neg_iwes), dim=1)
        else:
            return self.imager.create_iwe(warped, method='bilinear_vote', sigma=1,
                                          weight=weights)

    def calculate_smooth_loss(self, flow_lut, flow_to_next):
        if self.smooth_weight == 0:
            return torch.tensor(0., device=flow_lut.device)

        if self.smooth_type == 'on_flow_to_tref':
            flow_field = flow_lut
        elif self.smooth_type == 'on_flow_to_next':
            flow_field = flow_to_next
        else:
            raise ValueError

        flow_field = flow_field.permute(0, 1, 4, 5, 2, 3)
        _, _, _, c, h, w = flow_field.shape
        flow_field = flow_field.reshape(-1, c, h, w)
        return self.smooth_weight * utils.calculate_smoothness_loss(flow_field)

    def calculate_magn_loss(self, coeffs):
        if self.magn_weight == 0:
            return torch.tensor(0., device=coeffs.device)

        return self.magn_weight * torch.mean(torch.abs(coeffs))

    def calculate_scale_loss(self, coeff_grid):
        if self.scale_weight == 0:
            return torch.tensor(0., device=coeff_grid.device)

        if coeff_grid.shape[1] > 1:
            scale_sum_1 = torch.mean(torch.abs(coeff_grid[:, 1]))
            scale_sum_2 = torch.mean(torch.abs(coeff_grid[:, 2]))
        else:
            scale_sum_1 = torch.zeros(1, device=coeff_grid.device)
            scale_sum_2 = torch.zeros(1, device=coeff_grid.device)

        return self.scale_weight * (4 * scale_sum_2 + scale_sum_1)