import numpy as np


def make_continuous_copy(alpha):
    alpha = (alpha + np.pi) % (2.0 * np.pi) - np.pi
    continuous_x = np.zeros_like(alpha)
    continuous_x[0] = alpha[0]
    for i in range(1, len(alpha)):
        if not (np.sign(alpha[i]) == np.sign(alpha[i - 1])) and np.abs(alpha[i]) > np.pi / 2:
            continuous_x[i] = continuous_x[i - 1] + (
                    alpha[i] - alpha[i - 1]) - np.sign(
                (alpha[i] - alpha[i - 1])) * 2 * np.pi
        else:
            continuous_x[i] = continuous_x[i - 1] + (alpha[i] - alpha[i - 1])

    return continuous_x


def derivative_of(x, dt=1, radian=False):
    if radian:
        x = make_continuous_copy(x)

    not_nan_mask = ~np.isnan(x)
    masked_x = x[not_nan_mask]

    if masked_x.shape[-1] < 2:
        return np.zeros_like(x)

    dx = np.full_like(x, np.nan)
    dx[not_nan_mask] = np.ediff1d(masked_x, to_begin=(masked_x[1] - masked_x[0])) / dt

    return dx

