import torch
import torch.nn.functional as F

def calculate_focus_loss(iwes: torch.Tensor, loss_type: str = 'variance',
                         norm: str = 'l2'):
    if loss_type == 'variance':
        val = calculate_image_variance(iwes)
    elif loss_type == 'gradient_magnitude':
        val = calculate_gradient_magnitude(iwes, norm=norm)
    else:
        raise ValueError
    return 1 / val

def calculate_image_variance(iwes):
    variances = torch.var(iwes, dim=(-2, -1))
    return torch.mean(variances)

def calculate_gradient_magnitude(iwes, norm='l2'):
    if len(iwes.shape) == 3:
        iwes = iwes[:, None]
    dx, dy = gradient(iwes)
    if norm == 'l2':
        return torch.mean(torch.square(dx) + torch.square(dy))
    elif norm == 'l1':
        return torch.mean(torch.abs(dx) + torch.abs(dy))
    else:
        raise ValueError

def calculate_smoothness_loss(flow: torch.Tensor):
    """Regularization loss for learning flow fields
    as in Zhu19 (https://arxiv.org/abs/1812.08156)

    Args:
    flow: [b, 2, h, w]

    Returns:
    loss
    """
    dx, dy = gradient(flow)
    loss = (
        charbonnier_loss(dx)
      + charbonnier_loss(dy) 
    ) / 2.
    return loss

def charbonnier_loss(x: torch.Tensor, epsilon=1e-3):
    """
    Args:
    x: [b, ]
    epsilon: Small constant for numerical stability.

    Returns:
    Loss value as a PyTorch tensor.
    """
    loss = torch.sqrt(x ** 2 + epsilon ** 2)
    return torch.mean(loss)

def gradient(a: torch.Tensor):
    """
    Calculates the gradient of an input tensor a using convolution.

    Args:
    x: (batch_size, channels, height, width).

    Returns:
    The gradient of the input tensor.
    """
    kernel_x = torch.tensor([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]],
                             dtype=a.dtype,
                             device=a.device).view(1, 1, 3, 3)

    kernel_y = torch.tensor([[-1, -2, -1],
                             [ 0,  0,  0],
                             [ 1,  2,  1]],
                             dtype=a.dtype,
                             device=a.device).view(1, 1, 3, 3)

    channels = a.shape[1]
    kernel_x = kernel_x.repeat(channels, 1, 1, 1)
    kernel_y = kernel_y.repeat(channels, 1, 1, 1)

    grad_x = F.conv2d(a, kernel_x, padding=1, groups=channels)
    grad_y = F.conv2d(a, kernel_y, padding=1, groups=channels)

    return grad_x, grad_y