import torch as th

from .base import CurveBase
from .bezier import BezierCurves
from .polynomial import PolynomialCurves
from .learned import LearnedCurves


class CurveFactory:
    def __init__(self, basis_network=None) -> None:
        self.basis_network = basis_network

    def get_curve(self, curve_type, params):
        if curve_type == 'BEZIER':
            return BezierCurves(params)
        elif curve_type == 'POLYNOMIAL':
            return PolynomialCurves(params)
        elif curve_type == 'LEARNED':
            return LearnedCurves(params, self.basis_network)
        else:
            raise ValueError("Unsupported dataset type")

    def create_from_voxel_grid(self, curve_type,
                               voxel_grid: th.Tensor,
                               downsample_factor: int=8,
                               bezier_degree: int=2):
        if curve_type == 'BEZIER':
            curve = BezierCurves
        elif curve_type == 'POLYNOMIAL':
            curve = PolynomialCurves
        elif curve_type == 'LEARNED':
            return LearnedCurves.create_from_voxel_grid(voxel_grid, downsample_factor=downsample_factor,
                                                        bezier_degree=bezier_degree,
                                                        basis_network=self.basis_network)
        else:
            raise ValueError("Unsupported dataset type")

        return curve.create_from_voxel_grid(voxel_grid,
                                            downsample_factor=downsample_factor,
                                            bezier_degree=bezier_degree)
