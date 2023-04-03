import sys
import os
import dill
import json
import argparse
import torch
import numpy as np
import pandas as pd

sys.path.append("../../trajectron")
from tqdm import tqdm
from model.model_registrar import ModelRegistrar
from model.trajectron import Trajectron
import evaluation

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="model full path", type=str)
parser.add_argument("--checkpoint", help="model checkpoint to evaluate", type=int)
parser.add_argument("--data", help="full path to data file", type=str)
parser.add_argument("--output_path", help="path to output csv file", type=str)
parser.add_argument("--output_tag", help="name tag for output file", type=str)
parser.add_argument("--node_type", help="node type to evaluate", type=str)
args = parser.parse_args()


def load_model(model_dir, env, ts=100):
    model_registrar = ModelRegistrar(model_dir, 'cpu')
    model_registrar.load_models(ts)
    with open(os.path.join(model_dir, 'config.json'), 'r') as config_json:
        hyperparams = json.load(config_json)

    trajectron = Trajectron(model_registrar, hyperparams, None, 'cpu')

    trajectron.set_environment(env)
    trajectron.set_annealing_params()
    return trajectron, hyperparams


if __name__ == "__main__":
    with open(args.data, 'rb') as f:
        env = dill.load(f, encoding='latin1')

    eval_stg, hyperparams = load_model(args.model, env, ts=args.checkpoint)

    if 'override_attention_radius' in hyperparams:
        for attention_radius_override in hyperparams['override_attention_radius']:
            node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
            env.attention_radius[(node_type1, node_type2)] = float(attention_radius)

    scenes = env.scenes

    print("-- Preparing Node Graph")
    for scene in tqdm(scenes):
        scene.calculate_scene_graph(env.attention_radius,
                                    hyperparams['edge_addition_filter'],
                                    hyperparams['edge_removal_filter'])

    ph = hyperparams['prediction_horizon']
    max_hl = hyperparams['maximum_history_length']

    with torch.no_grad():
        ############### MOST LIKELY ###############
        eval_ade_batch_errors = np.array([])
        eval_fde_batch_errors = np.array([])
        print("-- Evaluating GMM Grid Sampled (Most Likely)")
        for i, scene in enumerate(scenes):
            print(f"---- Evaluating Scene {i + 1}/{len(scenes)}")
            timesteps = np.arange(scene.timesteps)
            predictions = eval_stg.predict(scene,
                                           timesteps,
                                           ph,
                                           num_samples=1,
                                           min_history_timesteps=7,
                                           min_future_timesteps=12,
                                           z_mode=False,
                                           gmm_mode=True,
                                           full_dist=True)  # This will trigger grid sampling

            batch_error_dict = evaluation.compute_batch_statistics(predictions,
                                                                   scene.dt,
                                                                   max_hl=max_hl,
                                                                   ph=ph,
                                                                   node_type_enum=env.NodeType,
                                                                   map=None,
                                                                   prune_ph_to_future=True,
                                                                   kde=False)

            eval_ade_batch_errors = np.hstack((eval_ade_batch_errors, batch_error_dict[args.node_type]['ade']))
            eval_fde_batch_errors = np.hstack((eval_fde_batch_errors, batch_error_dict[args.node_type]['fde']))
            total_number_testing_samples = eval_fde_batch_errors.shape[0]
            print('All         (ADE/FDE): %.2f/ %.2f   --- %d' % (
                eval_ade_batch_errors.mean(),
                eval_fde_batch_errors.mean(),
                total_number_testing_samples))
                
        print(np.mean(eval_fde_batch_errors))
        pd.DataFrame({'value': eval_ade_batch_errors, 'metric': 'ade', 'type': 'ml'}
                     ).to_csv(os.path.join(args.output_path, args.output_tag + '_ade_most_likely.csv'))
        pd.DataFrame({'value': eval_fde_batch_errors, 'metric': 'fde', 'type': 'ml'}
                     ).to_csv(os.path.join(args.output_path, args.output_tag + '_fde_most_likely.csv'))


        ############### MODE Z ###############
        eval_ade_batch_errors = np.array([])
        eval_fde_batch_errors = np.array([])
        eval_kde_nll = np.array([])
        print("-- Evaluating Mode Z")
        for i, scene in enumerate(scenes):
            print(f"---- Evaluating Scene {i+1}/{len(scenes)}")
            for t in tqdm(range(0, scene.timesteps, 10)):
                timesteps = np.arange(t, t + 10)
                predictions = eval_stg.predict(scene,
                                               timesteps,
                                               ph,
                                               num_samples=2000,
                                               min_history_timesteps=7,
                                               min_future_timesteps=12,
                                               z_mode=True,
                                               full_dist=False)

                if not predictions:
                    continue

                batch_error_dict = evaluation.compute_batch_statistics(predictions,
                                                                       scene.dt,
                                                                       max_hl=max_hl,
                                                                       ph=ph,
                                                                       node_type_enum=env.NodeType,
                                                                       map=None,
                                                                       prune_ph_to_future=True)
                eval_ade_batch_errors = np.hstack((eval_ade_batch_errors, batch_error_dict[args.node_type]['ade']))
                eval_fde_batch_errors = np.hstack((eval_fde_batch_errors, batch_error_dict[args.node_type]['fde']))
                eval_kde_nll = np.hstack((eval_kde_nll, batch_error_dict[args.node_type]['kde']))

        pd.DataFrame({'value': eval_ade_batch_errors, 'metric': 'ade', 'type': 'z_mode'}
                     ).to_csv(os.path.join(args.output_path, args.output_tag + '_ade_z_mode.csv'))
        pd.DataFrame({'value': eval_fde_batch_errors, 'metric': 'fde', 'type': 'z_mode'}
                     ).to_csv(os.path.join(args.output_path, args.output_tag + '_fde_z_mode.csv'))
        pd.DataFrame({'value': eval_kde_nll, 'metric': 'kde', 'type': 'z_mode'}
                     ).to_csv(os.path.join(args.output_path, args.output_tag + '_kde_z_mode.csv'))


        ############### BEST OF 20 ###############
        eval_ade_batch_errors = np.array([])
        eval_fde_batch_errors = np.array([])
        kalman_errors = np.array([])
        eval_kde_nll = np.array([])
        print("-- Evaluating best of 20")
        for i, scene in enumerate(scenes):
            print(f"---- Evaluating Scene {i + 1}/{len(scenes)}")
            for t in tqdm(range(0, scene.timesteps, 10)):
                timesteps = np.arange(t, t + 10)
                predictions = eval_stg.predict(scene,
                                               timesteps,
                                               ph,
                                               num_samples=20,
                                               min_history_timesteps=7,
                                               min_future_timesteps=12,
                                               z_mode=False,
                                               gmm_mode=False,
                                               full_dist=False)
                kalman_error = eval_stg.make_kalman(scene,
                                                timesteps,
                                                min_history_timesteps=7,
                                                min_future_timesteps=12)
                kalman_errors = np.hstack((kalman_errors, kalman_error))

                if not predictions:
                    continue

                batch_error_dict = evaluation.compute_batch_statistics(predictions,
                                                                       scene.dt,
                                                                       max_hl=max_hl,
                                                                       ph=ph,
                                                                       node_type_enum=env.NodeType,
                                                                       map=None,
                                                                       best_of=True,
                                                                       prune_ph_to_future=True)
                eval_ade_batch_errors = np.hstack((eval_ade_batch_errors, batch_error_dict[args.node_type]['ade']))
                eval_fde_batch_errors = np.hstack((eval_fde_batch_errors, batch_error_dict[args.node_type]['fde']))
                eval_kde_nll = np.hstack((eval_kde_nll, batch_error_dict[args.node_type]['kde']))
            total_number_testing_samples = eval_fde_batch_errors.shape[0]
            print('All         (ADE/FDE): %.2f/ %.2f   --- %d' % (
                eval_ade_batch_errors.mean(),
                eval_fde_batch_errors.mean(),
                total_number_testing_samples))
            assert kalman_errors.shape[0] == eval_fde_batch_errors.shape[0]
            largest_errors_indexes = np.argsort(kalman_errors)
            mask = np.ones(eval_ade_batch_errors.shape, dtype=bool)
            for top_index in range(1, 4):
                challenging = largest_errors_indexes[-int(
                    total_number_testing_samples * top_index / 100):]
                fde_errors_challenging = np.copy(eval_fde_batch_errors)
                ade_errors_challenging = np.copy(eval_ade_batch_errors)
                mask[challenging] = False
                fde_errors_challenging[mask] = 0
                ade_errors_challenging[mask] = 0
                print('Challenging Top %d (ADE/FDE): %.2f/ %.2f   --- %d' %
                        (top_index,
                        np.sum(ade_errors_challenging) / len(challenging),
                        np.sum(fde_errors_challenging) / len(challenging),
                        len(challenging)))        
        pd.DataFrame({'value': eval_ade_batch_errors, 'metric': 'ade', 'type': 'best_of'}
                     ).to_csv(os.path.join(args.output_path, args.output_tag + '_ade_best_of.csv'))
        pd.DataFrame({'value': eval_fde_batch_errors, 'metric': 'fde', 'type': 'best_of'}
                     ).to_csv(os.path.join(args.output_path, args.output_tag + '_fde_best_of.csv'))
        pd.DataFrame({'value': eval_kde_nll, 'metric': 'kde', 'type': 'best_of'}
                     ).to_csv(os.path.join(args.output_path, args.output_tag + '_kde_best_of.csv'))

        ############### FULL ###############
        eval_ade_batch_errors = np.array([])
        eval_fde_batch_errors = np.array([])
        eval_kde_nll = np.array([])
        print("-- Evaluating Full")
        for i, scene in enumerate(scenes):
            print(f"---- Evaluating Scene {i + 1}/{len(scenes)}")
            for t in tqdm(range(0, scene.timesteps, 10)):
                timesteps = np.arange(t, t + 10)
                predictions = eval_stg.predict(scene,
                                               timesteps,
                                               ph,
                                               num_samples=2000,
                                               min_history_timesteps=7,
                                               min_future_timesteps=12,
                                               z_mode=False,
                                               gmm_mode=False,
                                               full_dist=False)

                if not predictions:
                    continue

                batch_error_dict = evaluation.compute_batch_statistics(predictions,
                                                                       scene.dt,
                                                                       max_hl=max_hl,
                                                                       ph=ph,
                                                                       node_type_enum=env.NodeType,
                                                                       map=None,
                                                                       prune_ph_to_future=True)

                eval_ade_batch_errors = np.hstack((eval_ade_batch_errors, batch_error_dict[args.node_type]['ade']))
                eval_fde_batch_errors = np.hstack((eval_fde_batch_errors, batch_error_dict[args.node_type]['fde']))
                eval_kde_nll = np.hstack((eval_kde_nll, batch_error_dict[args.node_type]['kde']))

        pd.DataFrame({'value': eval_ade_batch_errors, 'metric': 'ade', 'type': 'full'}
                     ).to_csv(os.path.join(args.output_path, args.output_tag + '_ade_full.csv'))
        pd.DataFrame({'value': eval_fde_batch_errors, 'metric': 'fde', 'type': 'full'}
                     ).to_csv(os.path.join(args.output_path, args.output_tag + '_fde_full.csv'))
        pd.DataFrame({'value': eval_kde_nll, 'metric': 'kde', 'type': 'full'}
                     ).to_csv(os.path.join(args.output_path, args.output_tag + '_kde_full.csv'))
