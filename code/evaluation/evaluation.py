import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import binary_dilation
from scipy.stats import gaussian_kde
from utils import prediction_output_to_trajectories


def compute_mse(predicted_trajs, gt_traj):
    error = np.linalg.norm(predicted_trajs - gt_traj, axis=-1)
    mse = np.mean(error, axis=-1)
    return mse


def compute_fde(predicted_trajs, gt_traj):
    final_error = np.linalg.norm(predicted_trajs[:, -1] - gt_traj[-1], axis=-1)
    return final_error


def compute_kde_nll(predicted_trajs, gt_traj):
    kde_ll = 0.
    log_pdf_lower_bound = -20
    num_timesteps = gt_traj.shape[0]

    for timestep in range(num_timesteps):
        kde = gaussian_kde(predicted_trajs[:, timestep].T)
        pdf = np.clip(kde.logpdf(gt_traj[timestep].T), a_min=log_pdf_lower_bound, a_max=None)[0]
        kde_ll += pdf / num_timesteps

    return -kde_ll


def compute_obs_violations(predicted_trajs, map):
    obs_map = map.data

    interp_obs_map = RectBivariateSpline(range(obs_map.shape[1]),
                                         range(obs_map.shape[0]),
                                         binary_dilation(obs_map.T, iterations=4),
                                         kx=1, ky=1)

    old_shape = predicted_trajs.shape
    pred_trajs_map = map.to_map_points(predicted_trajs.reshape((-1, 2)))

    traj_obs_values = interp_obs_map(pred_trajs_map[:, 0], pred_trajs_map[:, 1], grid=False)
    traj_obs_values = traj_obs_values.reshape((old_shape[0], old_shape[1]))
    num_viol_trajs = np.sum(traj_obs_values.max(axis=1) > 0, dtype=float)

    return num_viol_trajs


def compute_batch_statistics(prediction_output_dict, dt, max_hl, ph, kde=True,
                             map=None, prune_ph_to_future=False, best_of=False):

    (prediction_dict,
     _,
     futures_dict) = prediction_output_to_trajectories(prediction_output_dict,
                                                       dt,
                                                       max_hl,
                                                       ph,
                                                       prune_ph_to_future=prune_ph_to_future)

    batch_error_dict = {'mse': list(), 'fde': list(), 'kde': list(), 'obs_viols': list()}

    for t in prediction_dict.keys():
        for node in prediction_dict[t].keys():
            mse_errors = compute_mse(prediction_dict[t][node], futures_dict[t][node])
            fde_errors = compute_fde(prediction_dict[t][node], futures_dict[t][node])
            if kde:
                kde_ll = compute_kde_nll(prediction_dict[t][node], futures_dict[t][node])
            else:
                kde_ll = 0
            if map is not None:
                obs_viols = compute_obs_violations(prediction_dict[t][node], map)
            else:
                obs_viols = 0
            if best_of:
                mse_errors = np.min(mse_errors)
                fde_errors = np.min(fde_errors)
                kde_ll = np.min(kde_ll)
            batch_error_dict['mse'].append(mse_errors)
            batch_error_dict['fde'].append(fde_errors)
            batch_error_dict['kde'].append(kde_ll)
            batch_error_dict['obs_viols'].append(obs_viols)

    return (np.hstack(batch_error_dict['mse']),
            np.hstack(batch_error_dict['fde']),
            np.hstack(batch_error_dict['kde']),
            np.hstack(batch_error_dict['obs_viols']))