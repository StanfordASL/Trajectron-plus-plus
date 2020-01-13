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

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="model full path", type=str)
parser.add_argument("--checkpoint", help="model checkpoint to evaluate", type=int)
parser.add_argument("--data", help="full path to data file", type=str)
parser.add_argument("--output", help="full path to output csv file", type=str)
args = parser.parse_args()


def load_model(model_dir, env, ts=99):
    model_registrar = ModelRegistrar(model_dir, 'cpu')
    model_registrar.load_models(ts)
    with open(os.path.join(model_dir, 'config.json'), 'r') as config_json:
        hyperparams = json.load(config_json)

    stg = SpatioTemporalGraphCVAEModel(None, model_registrar,
                                       hyperparams,
                                       None, 'cpu')

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
                                    hyperparams['dims'],
                                    hyperparams['edge_addition_filter'],
                                    hyperparams['edge_removal_filter'])


    ph = hyperparams['prediction_horizon']
    max_hl = hyperparams['maximum_history_length']

    with torch.no_grad():
        eval_ade_batch_errors = np.array([])
        eval_fde_batch_errors = np.array([])
        eval_kde_nll = np.array([])
        print("-- Evaluating Full")
        for i, scene in enumerate(scenes):
            print(f"---- Evaluating Scene {i+1}/{len(scenes)}")
            for t in tqdm(range(0, scene.timesteps, 10)):
                timestep = np.arange(t, t + 10)
                predictions = eval_stg.predict(scene,
                                               timestep,
                                               ph,
                                               num_samples_z=2000,
                                               most_likely_z=False,
                                               min_future_timesteps=1)

                if not predictions:
                    continue

                (eval_ade_batch_errors_s,
                 eval_fde_batch_errors_s,
                 eval_kde_nll_s,
                 _) = evaluation.compute_batch_statistics(predictions,
                                                          scene.dt,
                                                          max_hl=max_hl,
                                                          ph=ph,
                                                          map=None,
                                                          prune_ph_to_future=True)
                eval_ade_batch_errors = np.hstack((eval_ade_batch_errors, eval_ade_batch_errors_s))
                eval_fde_batch_errors = np.hstack((eval_fde_batch_errors, eval_fde_batch_errors_s))
                eval_kde_nll = np.hstack((eval_kde_nll, eval_kde_nll_s))

        print(np.mean(eval_fde_batch_errors))
        pd.DataFrame({'error_value': eval_ade_batch_errors, 'error_type': 'ade', 'type': 'full'}).to_csv(args.output + '_ade_full.csv')
        pd.DataFrame({'error_value': eval_fde_batch_errors, 'error_type': 'fde', 'type': 'full'}).to_csv(args.output + '_fde_full.csv')
        pd.DataFrame({'error_value': eval_kde_nll, 'error_type': 'kde', 'type': 'full'}).to_csv(args.output + '_kde_full.csv')

        print(np.mean(eval_fde_batch_errors))
        eval_ade_batch_errors = np.array([])
        eval_fde_batch_errors = np.array([])
        eval_kde_nll = np.array([])
        print("-- Evaluating most likely Z")
        for i, scene in enumerate(scenes):
            print(f"---- Evaluating Scene {i+1}/{len(scenes)}")
            for t in tqdm(range(0, scene.timesteps, 10)):
                timestep = np.arange(t, t + 10)
                predictions = eval_stg.predict(scene,
                                               timestep,
                                               ph,
                                               num_samples_z=2000,
                                               most_likely_z=True,
                                               min_future_timesteps=1)

                if not predictions:
                    continue

                (eval_ade_batch_errors_s,
                 eval_fde_batch_errors_s,
                 eval_kde_nll_s,
                 _) = evaluation.compute_batch_statistics(predictions,
                                                          scene.dt,
                                                          max_hl=max_hl,
                                                          ph=ph,
                                                          map=None,
                                                          prune_ph_to_future=True)
                eval_ade_batch_errors = np.hstack((eval_ade_batch_errors, eval_ade_batch_errors_s))
                eval_fde_batch_errors = np.hstack((eval_fde_batch_errors, eval_fde_batch_errors_s))
                eval_kde_nll = np.hstack((eval_kde_nll, eval_kde_nll_s))

        pd.DataFrame({'error_value': eval_ade_batch_errors, 'error_type': 'ade', 'type': 'z_best'}).to_csv(args.output + '_ade_z_best.csv')
        pd.DataFrame({'error_value': eval_fde_batch_errors, 'error_type': 'fde', 'type': 'z_best'}).to_csv(args.output + '_fde_z_best.csv')
        pd.DataFrame({'error_value': eval_kde_nll, 'error_type': 'kde', 'type': 'z_best'}).to_csv(args.output + '_kde_z_best.csv')

        eval_ade_batch_errors = np.array([])
        eval_fde_batch_errors = np.array([])
        eval_kde_nll = np.array([])
        print("-- Evaluating most likely Z and GMM")
        for i, scene in enumerate(scenes):
            print(f"---- Evaluating Scene {i+1}/{len(scenes)}")
            timesteps = scene.sample_timesteps(scene.timesteps)

            predictions = eval_stg.predict(scene,
                                           timesteps,
                                           ph,
                                           num_samples_z=1,
                                           most_likely_z=True,
                                           most_likely_gmm=True,
                                           min_future_timesteps=1)

            (eval_ade_batch_errors_s,
             eval_fde_batch_errors_s,
             eval_kde_nll_s,
             _) = evaluation.compute_batch_statistics(predictions,
                                                      scene.dt,
                                                      max_hl=max_hl,
                                                      ph=ph,
                                                      map=None,
                                                      kde=False,
                                                      prune_ph_to_future=True)
            eval_ade_batch_errors = np.hstack((eval_ade_batch_errors, eval_ade_batch_errors_s))
            eval_fde_batch_errors = np.hstack((eval_fde_batch_errors, eval_fde_batch_errors_s))

        pd.DataFrame({'error_value': eval_ade_batch_errors, 'error_type': 'ade', 'type': 'mm'}).to_csv(args.output + '_ade_mm.csv')
        pd.DataFrame({'error_value': eval_fde_batch_errors, 'error_type': 'fde', 'type': 'mm'}).to_csv(args.output + '_fde_mm.csv')

        eval_ade_batch_errors = np.array([])
        eval_fde_batch_errors = np.array([])
        eval_kde_nll = np.array([])
        print("-- Evaluating best of 20")
        for i, scene in enumerate(scenes):
            print(f"---- Evaluating Scene {i}/{len(scenes)}")
            for t in tqdm(range(0, scene.timesteps, 10)):
                timestep = np.arange(t, t + 10)
                predictions = eval_stg.predict(scene,
                                               timestep,
                                               ph,
                                               num_samples_z=20,
                                               most_likely_z=False,
                                               min_future_timesteps=1)

                if not predictions:
                    continue

                (eval_ade_batch_errors_s,
                 eval_fde_batch_errors_s,
                 eval_kde_nll_s,
                 _) = evaluation.compute_batch_statistics(predictions,
                                                          scene.dt,
                                                          max_hl=max_hl,
                                                          ph=ph,
                                                          map=None,
                                                          best_of=True,
                                                          prune_ph_to_future=True)
                eval_ade_batch_errors = np.hstack((eval_ade_batch_errors, eval_ade_batch_errors_s))
                eval_fde_batch_errors = np.hstack((eval_fde_batch_errors, eval_fde_batch_errors_s))
                eval_kde_nll = np.hstack((eval_kde_nll, eval_kde_nll_s))

        pd.DataFrame({'error_value': eval_ade_batch_errors, 'error_type': 'ade', 'type': 'best_of'}).to_csv(args.output + '_ade_best_of.csv')
        pd.DataFrame({'error_value': eval_fde_batch_errors, 'error_type': 'fde', 'type': 'best_of'}).to_csv(args.output + '_fde_best_of.csv')
        pd.DataFrame({'error_value': eval_kde_nll, 'error_type': 'kde', 'type': 'best_of'}).to_csv(args.output + '_kde_best_of.csv')
