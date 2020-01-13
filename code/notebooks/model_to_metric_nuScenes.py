import sys
sys.path.append('../../code')
import os
import pickle
import json
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from model.model_registrar import ModelRegistrar
from model.dyn_stg import SpatioTemporalGraphCVAEModel
import evaluation
from utils import prediction_output_to_trajectories
from scipy.interpolate import RectBivariateSpline

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="model full path", type=str)
parser.add_argument("--checkpoint", help="model checkpoint to evaluate", type=int)
parser.add_argument("--data", help="full path to data file", type=str)
parser.add_argument("--output", help="full path to output csv file", type=str)
parser.add_argument("--node_type", help="Node Type to evaluate", type=str)
parser.add_argument("--prediction_horizon", nargs='+', help="prediction horizon", type=int, default=None)
args = parser.parse_args()


def compute_obs_violations(predicted_trajs, map):
    obs_map = 1 - map.fdata[..., 0]

    interp_obs_map = RectBivariateSpline(range(obs_map.shape[0]),
                                         range(obs_map.shape[1]),
                                         obs_map,
                                         kx=1, ky=1)

    old_shape = predicted_trajs.shape
    pred_trajs_map = map.to_map_points(predicted_trajs.reshape((-1, 2)))

    traj_obs_values = interp_obs_map(pred_trajs_map[:, 0], pred_trajs_map[:, 1], grid=False)
    traj_obs_values = traj_obs_values.reshape((old_shape[0], old_shape[1]))
    num_viol_trajs = np.sum(traj_obs_values.max(axis=1) > 0, dtype=float)

    return num_viol_trajs

def compute_heading_error(prediction_output_dict, dt, max_hl, ph, node_type_enum, kde=True, obs=False, map=None):

    heading_error = list()

    for t in prediction_output_dict.keys():
        for node in prediction_output_dict[t].keys():
            if node.type.name == 'VEHICLE':
                gt_vel = node.get(t + ph - 1, {'velocity': ['x', 'y']})[0]
                gt_heading = np.arctan2(gt_vel[1], gt_vel[0])
                our_heading = np.arctan2(prediction_output_dict[t][node][..., -2, 1], prediction_output_dict[t][node][..., -2,  0])
                he = np.mean(np.abs(gt_heading - our_heading)) % (2 * np.pi)
                heading_error.append(he)

    return heading_error


def load_model(model_dir, env, ts=99):
    model_registrar = ModelRegistrar(model_dir, 'cpu')
    model_registrar.load_models(ts)
    with open(os.path.join(model_dir, 'config.json'), 'r') as config_json:
        hyperparams = json.load(config_json)

    stg = SpatioTemporalGraphCVAEModel(model_registrar,
                                       hyperparams,
                                       None, 'cuda:0')
    hyperparams['incl_robot_node'] = False

    stg.set_scene_graph(env)
    stg.set_annealing_params()
    return stg, hyperparams


if __name__ == "__main__":
    with open(args.data, 'rb') as f:
        env = pickle.load(f, encoding='latin1')
    scenes = env.scenes

    eval_stg, hyperparams = load_model(args.model, env, ts=args.checkpoint)

    print("-- Preparing Node Graph")
    for scene in tqdm(scenes):
        scene.calculate_scene_graph(hyperparams['edge_radius'],
                                    hyperparams['state'],
                                    hyperparams['edge_addition_filter'],
                                    hyperparams['edge_removal_filter'])

    if args.prediction_horizon is None:
        args.prediction_horizon = [hyperparams['prediction_horizon']]

    for ph in args.prediction_horizon:
        print(f"Prediction Horizon: {ph}")
        max_hl = hyperparams['maximum_history_length']
        node_type = env.NodeType[args.node_type]
        print(f"Node Type: {node_type.name}")
        print(f"Edge Radius: {hyperparams['edge_radius']}")

        with torch.no_grad():
            eval_ade_batch_errors = np.array([])
            eval_fde_batch_errors = np.array([])
            eval_kde_nll = np.array([])
            eval_obs_viols = np.array([])
            print("-- Evaluating Full")
            for i, scene in enumerate(tqdm(scenes)):
                for timestep in range(scene.timesteps):
                    predictions = eval_stg.predict(scene,
                                                   np.array([timestep]),
                                                   ph,
                                                   num_samples_z=2000,
                                                   most_likely_z=False,
                                                   min_future_timesteps=8)

                    if not predictions:
                        continue

                    eval_error_dict = evaluation.compute_batch_statistics(predictions,
                                                                          scene.dt,
                                                                          node_type_enum=env.NodeType,
                                                                          max_hl=max_hl,
                                                                          ph=ph,
                                                                          map=scene.map[node_type.name],
                                                                          obs=True)

                    eval_ade_batch_errors = np.hstack((eval_ade_batch_errors, eval_error_dict[node_type]['ade']))
                    eval_fde_batch_errors = np.hstack((eval_fde_batch_errors, eval_error_dict[node_type]['fde']))
                    eval_kde_nll = np.hstack((eval_kde_nll, eval_error_dict[node_type]['kde']))
                    eval_obs_viols = np.hstack((eval_obs_viols, eval_error_dict[node_type]['obs_viols']))

                    del predictions
                    del eval_error_dict

            print(f"Final Mean Displacement Error @{ph * scene.dt}s: {np.mean(eval_fde_batch_errors)}")
            print(f"Road Violations @{ph * scene.dt}s: {100 * np.sum(eval_obs_viols) / (eval_obs_viols.shape[0] * 2000)}%")
            pd.DataFrame({'error_value': eval_ade_batch_errors, 'error_type': 'ade', 'type': 'full', 'ph': ph}).to_csv(args.output + '_ade_full_' + str(ph)+'ph' + '.csv')
            pd.DataFrame({'error_value': eval_fde_batch_errors, 'error_type': 'fde', 'type': 'full', 'ph': ph}).to_csv(args.output + '_fde_full' + str(ph)+'ph' + '.csv')
            pd.DataFrame({'error_value': eval_kde_nll, 'error_type': 'kde', 'type': 'full', 'ph': ph}).to_csv(args.output + '_kde_full' + str(ph)+'ph' + '.csv')
            pd.DataFrame({'error_value': eval_obs_viols, 'error_type': 'obs', 'type': 'full', 'ph': ph}).to_csv(args.output + '_obs_full' + str(ph)+'ph' + '.csv')

            eval_ade_batch_errors = np.array([])
            eval_fde_batch_errors = np.array([])
            eval_heading_err = np.array([])
            eval_obs_viols = np.array([])
            print("-- Evaluating most likely Z and GMM")
            for i, scene in enumerate(scenes):
                print(f"---- Evaluating Scene {i+1}/{len(scenes)}")
                for t in np.arange(0, scene.timesteps, 20):
                    timesteps = np.arange(t, t+20)

                    predictions = eval_stg.predict(scene,
                                                   timesteps,
                                                   ph,
                                                   num_samples_z=1,
                                                   most_likely_z=True,
                                                   most_likely_gmm=True,
                                                   min_future_timesteps=8)

                    eval_error_dict = evaluation.compute_batch_statistics(predictions,
                                                              scene.dt,
                                                                          node_type_enum=env.NodeType,
                                                              max_hl=max_hl,
                                                              ph=ph,
                                                              map=1 - scene.map[node_type.name].fdata[..., 0],
                                                              kde=False)
                    eval_ade_batch_errors = np.hstack((eval_ade_batch_errors, eval_error_dict[node_type]['ade']))
                    eval_fde_batch_errors = np.hstack((eval_fde_batch_errors, eval_error_dict[node_type]['fde']))
                    eval_obs_viols = np.hstack((eval_obs_viols, eval_error_dict[node_type]['obs_viols']))

                    heading_error = compute_heading_error(predictions,
                                                              scene.dt,
                                                            node_type_enum=env.NodeType,
                                                              max_hl=max_hl,
                                                              ph=ph,
                                                              map=1 - scene.map[node_type.name].fdata[..., 0],
                                                              kde=False)
                    eval_heading_err = np.hstack((eval_heading_err, heading_error))

            print(f"Final Displacement Error @{ph * scene.dt}s: {np.mean(eval_fde_batch_errors)}")
            pd.DataFrame({'error_value': eval_ade_batch_errors, 'error_type': 'ade', 'type': 'mm', 'ph': ph}).to_csv(args.output + '_ade_mm' + str(ph)+'ph' + '.csv')
            pd.DataFrame({'error_value': eval_fde_batch_errors, 'error_type': 'fde', 'type': 'mm', 'ph': ph}).to_csv(args.output + '_fde_mm' + str(ph)+'ph' + '.csv')
            pd.DataFrame({'error_value': eval_obs_viols, 'error_type': 'obs', 'type': 'mm', 'ph': ph}).to_csv( args.output + '_obs_mm' + str(ph)+'ph' + '.csv')
