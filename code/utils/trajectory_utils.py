import numpy as np
from scipy.integrate import cumtrapz


def integrate_trajectory(v, x0, dt):
    xd_ = cumtrapz(v[..., 0], axis=1, dx=dt) + x0[0]
    yd_ = cumtrapz(v[..., 1], axis=1, dx=dt) + x0[1]
    integrated = np.stack([xd_, yd_], axis=2)
    return integrated


def prediction_output_to_trajectories(prediction_output_dict,
                                      dt,
                                      max_h,
                                      ph,
                                      map=None,
                                      gmm_agg='mean',
                                      prune_ph_to_future=False):

    prediction_timesteps = prediction_output_dict.keys()

    output_dict = dict()
    histories_dict = dict()
    futures_dict = dict()

    for t in prediction_timesteps:
        histories_dict[t] = dict()
        output_dict[t] = dict()
        futures_dict[t] = dict()
        prediction_nodes = prediction_output_dict[t].keys()
        for node in prediction_nodes:
            predictions_output = prediction_output_dict[t][node]
            if node.is_robot:
                continue
            history = node.get(np.array([t - max_h, t]), ['position'], ['x', 'y'])  # History includes current pos
            history = history[~np.isnan(history.sum(axis=1))]

            future = node.get(np.array([t + 1, t + ph]), ['position'], ['x', 'y'])
            future = future[~np.isnan(future.sum(axis=1))]

            current_pos = node.get(t, ['position'], ['x', 'y'])[0]  # List with single item
            current_vel = node.get(t, ['velocity'], ['x', 'y'])[0]  # List with single item

            predictions_output = getattr(predictions_output, gmm_agg)(axis=1)

            if prune_ph_to_future:
                predictions_output = predictions_output[:, :future.shape[0]]
                if predictions_output.shape[1] == 0:
                    continue

            vel_broad = np.expand_dims(np.broadcast_to(current_vel,
                                                       (predictions_output.shape[0],
                                                        current_vel.shape[-1])), axis=-2)
            vel = np.concatenate((vel_broad, predictions_output), axis=1)

            if map is None:
                histories_dict[t][node] = history
                output_dict[t][node] = integrate_trajectory(vel, current_pos, dt=dt)
                futures_dict[t][node] = future
            else:
                histories_dict[t][node] = map.to_map_points(history)
                output_dict[t][node] = map.to_map_points(integrate_trajectory(vel, current_pos, dt=dt))
                futures_dict[t][node] = map.to_map_points(future)

    return output_dict, histories_dict, futures_dict
