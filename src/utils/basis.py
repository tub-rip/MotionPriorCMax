import numpy as np
import torch

def compute_basis(coeffs, times, num_basis, basis_type, basis_network=None):
    """Calculate trajectory values for given coefficients at n_t
    given times.

    Args:
        coeffs: [b, s, 2, n, k]
        times: [1, n_t]

    Raises:
        ValueError: _description_

    Returns:
        [b, n, k, n_t, 2]
    """
    if basis_type == "dct":
        K = num_basis
        T = 1
        k_idx = torch.arange(1, K+1, device=coeffs.device)
        A = (2 * times[..., None] + 1) * k_idx[None, None, :]
        in_cos = np.pi / (2.0 * T) * A
        basis = np.sqrt(2.0 / T) * torch.cos(in_cos)

    elif basis_type == "learned":
        basis = basis_network(times[..., None]) # [b, m, k]

    elif basis_type == "polynomial":
        k_idx = torch.arange(1, num_basis+1, device=coeffs.device)
        basis = times[..., None] ** k_idx[None, None, :]
    else:
        raise ValueError

    raw_coeff_y = coeffs[..., 0, :, :] # [b, n, k]
    raw_coeff_x = coeffs[..., 1, :, :] # [b, n, k]

    y_product = basis[..., None, :, :] * raw_coeff_y[..., None, :] # [n, m, k]
    x_product = basis[..., None, :, :] * raw_coeff_x[..., None, :] # [n, m, k]

    coords = torch.stack([
        torch.sum(y_product, dim=-1),
        torch.sum(x_product, dim=-1)
    ], dim=-1)

    return torch.sum(coords, dim=1) # Sum trajectories of all scales
