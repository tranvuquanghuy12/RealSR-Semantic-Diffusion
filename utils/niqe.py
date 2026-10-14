import os
import cv2
import math
import numpy as np
from scipy.ndimage import convolve
from scipy.special import gamma


def to_y_channel(img):
    """Change image to Y channel of YCbCr (MATLAB convention).
    Args:
        img (ndarray): Images with range [0, 255].
    Returns:
        (ndarray): Images with range [0, 255] (float type).
    """
    img = img.astype(np.float32) / 255.
    if img.ndim == 3 and img.shape[2] == 3:
        img = np.dot(img, [65.481, 128.553, 24.966]) + 16.0
    return img


def reorder_image(img, input_order='HWC'):
    """Reorder images to 'HWC' order."""
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
    if len(img.shape) == 2:
        img = img[..., None]
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    return img


def estimate_aggd_param(block):
    """Estimate AGGD (Asymmetric Generalized Gaussian Distribution) parameters."""
    block = block.flatten()
    gam = np.arange(0.2, 10.001, 0.001)  # len = 9801
    gam_reciprocal = np.reciprocal(gam)
    r_gam = np.square(gamma(gam_reciprocal * 2)) / (
        gamma(gam_reciprocal) * gamma(gam_reciprocal * 3))

    left_std = np.sqrt(np.mean(block[block < 0]**2))
    right_std = np.sqrt(np.mean(block[block > 0]**2))
    if right_std == 0:
        right_std = 1e-7
    gammahat = left_std / right_std
    mean_sq = np.mean(block**2)
    if mean_sq == 0:
        mean_sq = 1e-7
    rhat = (np.mean(np.abs(block)))**2 / mean_sq
    denom = ((gammahat**2 + 1)**2)
    if denom == 0:
        denom = 1e-7
    rhatnorm = (rhat * (gammahat**3 + 1) * (gammahat + 1)) / denom
    array_position = np.argmin((r_gam - rhatnorm)**2)

    alpha = gam[array_position]
    beta_l = left_std * np.sqrt(gamma(1 / alpha) / gamma(3 / alpha))
    beta_r = right_std * np.sqrt(gamma(1 / alpha) / gamma(3 / alpha))
    return (alpha, beta_l, beta_r)


def compute_feature(block):
    """Compute 18-dimensional features per block."""
    feat = []
    alpha, beta_l, beta_r = estimate_aggd_param(block)
    feat.extend([alpha, (beta_l + beta_r) / 2])

    shifts = [[0, 1], [1, 0], [1, 1], [1, -1]]
    for i in range(len(shifts)):
        shifted_block = np.roll(block, shifts[i], axis=(0, 1))
        alpha, beta_l, beta_r = estimate_aggd_param(block * shifted_block)
        mean = (beta_r - beta_l) * (gamma(2 / alpha) / gamma(1 / alpha))
        feat.extend([alpha, mean, beta_l, beta_r])
    return feat


def niqe(img, mu_pris_param, cov_pris_param, gaussian_window, block_size_h=96, block_size_w=96):
    """Calculate NIQE metric on a 2D gray or Y channel image."""
    assert img.ndim == 2, 'Input image must be a gray or Y (of YCbCr) image with shape (h, w).'
    h, w = img.shape
    num_block_h = math.floor(h / block_size_h)
    num_block_w = math.floor(w / block_size_w)
    if num_block_h == 0 or num_block_w == 0:
        # Fallback for very small images: resize to minimum block size
        h_pad = max(h, block_size_h)
        w_pad = max(w, block_size_w)
        img = cv2.resize(img, (w_pad, h_pad), interpolation=cv2.INTER_LINEAR)
        h, w = img.shape
        num_block_h = math.floor(h / block_size_h)
        num_block_w = math.floor(w / block_size_w)

    img = img[0:num_block_h * block_size_h, 0:num_block_w * block_size_w]

    distparam = []
    for scale in (1, 2):
        mu = convolve(img, gaussian_window, mode='nearest')
        sigma = np.sqrt(np.abs(convolve(np.square(img), gaussian_window, mode='nearest') - np.square(mu)))
        img_normalized = (img - mu) / (sigma + 1)

        feat = []
        for idx_w in range(num_block_w):
            for idx_h in range(num_block_h):
                block = img_normalized[
                    idx_h * block_size_h // scale:(idx_h + 1) * block_size_h // scale,
                    idx_w * block_size_w // scale:(idx_w + 1) * block_size_w // scale
                ]
                feat.append(compute_feature(block))

        distparam.append(np.array(feat))
        if scale == 1:
            h_curr, w_curr = img.shape
            img = cv2.resize(img / 255., (max(w_curr // 2, 1), max(h_curr // 2, 1)), interpolation=cv2.INTER_LINEAR)
            img = img * 255.

    distparam = np.concatenate(distparam, axis=1)

    mu_distparam = np.nanmean(distparam, axis=0)
    distparam_no_nan = distparam[~np.isnan(distparam).any(axis=1)]
    if len(distparam_no_nan) < 2:
        return float('nan')
    cov_distparam = np.cov(distparam_no_nan, rowvar=False)

    invcov_param = np.linalg.pinv((cov_pris_param + cov_distparam) / 2)
    diff = mu_pris_param - mu_distparam
    quality = np.matmul(np.matmul(diff, invcov_param), np.transpose(diff))
    quality = np.sqrt(max(float(np.squeeze(quality)), 0.0))
    return float(quality)


def calculate_niqe(img, crop_border=0, input_order='HWC', convert_to='y'):
    """Calculate NIQE (Natural Image Quality Evaluator) metric.
    Args:
        img (ndarray): Input image in range [0, 255] with float or uint8 type.
        crop_border (int): Cropped pixels on each edge.
        input_order (str): 'HWC', 'CHW', or 'HW'.
        convert_to (str): 'y' (YCbCr luminance) or 'gray'.
    Returns:
        float: NIQE score (lower indicates better naturalness).
    """
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    params_path = os.path.join(curr_dir, 'niqe_pris_params.npz')
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"niqe_pris_params.npz not found in {curr_dir}")

    niqe_pris_params = np.load(params_path)
    mu_pris_param = niqe_pris_params['mu_pris_param']
    cov_pris_param = niqe_pris_params['cov_pris_param']
    gaussian_window = niqe_pris_params['gaussian_window']

    img = img.astype(np.float32)
    if input_order != 'HW':
        img = reorder_image(img, input_order=input_order)
        if convert_to == 'y':
            img = to_y_channel(img)
        elif convert_to == 'gray':
            img = cv2.cvtColor(img / 255., cv2.COLOR_BGR2GRAY) * 255.
        img = np.squeeze(img)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border]

    return niqe(img, mu_pris_param, cov_pris_param, gaussian_window)
