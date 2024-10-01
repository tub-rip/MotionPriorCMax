import numpy as np
import torch
import cv2

def visualize_optical_flow(flow):
     flow = torch.squeeze(flow)
     if type(flow) == torch.Tensor:
        flow = flow.cpu().detach().numpy()
     y = flow[0]
     x = flow[1]
     rgb_flow, color_wheel, _ = color_optical_flow(y, x)
     return rgb_flow.transpose(2, 0, 1), color_wheel.transpose(2, 0, 1)

def color_optical_flow(
        flow_y: np.ndarray, flow_x: np.ndarray, max_magnitude=None, ord=1.0
    ):
        """Color optical flow.
        Args:
            flow_y (numpy.ndarray) ... [H x W], height direction.
            flow_x (numpy.ndarray) ... [H x W], width direction.
            max_magnitude (float, optional) ... Max magnitude used for the colorization. Defaults to None.
            ord (float) ... 1: our usual, 0.5: DSEC colorinzing.

        Returns:
            flow_rgb (np.ndarray) ... [W, H]
            color_wheel (np.ndarray) ... [H, H] color wheel
            max_magnitude (float) ... max magnitude of the flow.
        """
        flows = np.stack((flow_y, flow_x), axis=2)
        flows[np.isinf(flows)] = 0
        flows[np.isnan(flows)] = 0
        mag = np.linalg.norm(flows, axis=2) ** ord
        ang = (np.arctan2(flow_x, flow_y) + np.pi) * 180.0 / np.pi / 2.0
        ang = ang.astype(np.uint8)
        hsv = np.zeros([flow_y.shape[0], flow_y.shape[1], 3], dtype=np.uint8)
        hsv[:, :, 0] = ang
        hsv[:, :, 1] = 255
        if max_magnitude is None:
            max_magnitude = mag.max()
        hsv[:, :, 2] = (255 * mag / (max_magnitude + 1e-6)).astype(np.uint8)
        flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        # Color wheel
        hsv = np.zeros([flow_y.shape[0], flow_y.shape[0], 3], dtype=np.uint8)
        xx, yy = np.meshgrid(
            np.linspace(-1, 1, flow_y.shape[0]), np.linspace(-1, 1, flow_y.shape[0])
        )
        mag = np.linalg.norm(np.stack((xx, yy), axis=2), axis=2)
        ang = (np.arctan2(xx, yy) + np.pi) * 180 / np.pi / 2.0
        hsv[:, :, 0] = ang.astype(np.uint8)
        hsv[:, :, 1] = 255
        hsv[:, :, 2] = (255 * mag / mag.max()).astype(np.uint8)
        color_wheel = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        return flow_rgb, color_wheel, max_magnitude

def normalize_images(images, invert=False):
    min_values = images.min(axis=(1, 2), keepdims=True)
    max_values = images.max(axis=(1, 2), keepdims=True)
    normalized_images = 255 * (images - min_values) / (max_values - min_values + 1e-6)
    if invert:
        normalized_images = 255 - normalized_images
    return normalized_images.astype(np.uint8)