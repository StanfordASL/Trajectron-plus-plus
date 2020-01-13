import torch
from torch import nn, optim
import numpy as np
import os
import time
import psutil
import pickle
import json
import random
import argparse
import pathlib
import visualization
import evaluation
import matplotlib.pyplot as plt
from model.dyn_stg import SpatioTemporalGraphCVAEModel
from model.model_registrar import ModelRegistrar
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser()
parser.add_argument("--conf", help="path to json config file for hyperparameters",
                    type=str, default='config.json')
parser.add_argument("--offline_scene_graph", help="whether to precompute the scene graphs offline, options are 'no' and 'yes'",
                    type=str, default='yes')
parser.add_argument("--dynamic_edges", help="whether to use dynamic edges or not, options are 'no' and 'yes'",
                    type=str, default='yes')
parser.add_argument("--edge_radius", help="the radius (in meters) within which two nodes will be connected by an edge",
                    type=float, default=3.0)
parser.add_argument("--edge_state_combine_method", help="the method to use for combining edges of the same type",
                    type=str, default='sum')
parser.add_argument("--edge_influence_combine_method", help="the method to use for combining edge influences",
                    type=str, default='attention')
parser.add_argument('--edge_addition_filter', nargs='+', help="what scaling to use for edges as they're created",
                    type=float, default=[0.25, 0.5, 0.75, 1.0]) # We automatically pad left with 0.0
parser.add_argument('--edge_removal_filter', nargs='+', help="what scaling to use for edges as they're removed",
                    type=float, default=[1.0, 0.0]) # We automatically pad right with 0.0
parser.add_argument('--incl_robot_node', help="whether to include a robot node in the graph or simply model all agents",
                    action='store_true')
parser.add_argument('--use_map_encoding', help="Whether to use map encoding or not",
                    action='store_true')

parser.add_argument("--data_dir", help="what dir to look in for data",
                    type=str, default='../data/processed')
parser.add_argument("--train_data_dict", help="what file to load for training data",
                    type=str, default='eth_train.pkl')
parser.add_argument("--eval_data_dict", help="what file to load for evaluation data",
                    type=str, default='eth_val.pkl')
parser.add_argument("--log_dir", help="what dir to save training information (i.e., saved models, logs, etc)",
                    type=str, default='../data/eth_univ/logs')

parser.add_argument('--device', help='what device to perform training on',
                    type=str, default='cuda:1')
parser.add_argument("--eval_device", help="what device to use during evaluation",
                    type=str, default=None)

parser.add_argument("--num_iters", help="number of iterations to train for",
                    type=int, default=2000)
parser.add_argument('--batch_multiplier', help='how many minibatches to run per iteration of training',
                    type=int, default=1)
parser.add_argument('--batch_size', help='training batch size',
                    type=int, default=256)
parser.add_argument('--eval_batch_size', help='evaluation batch size',
                    type=int, default=256)
parser.add_argument('--k_eval', help='how many samples to take during evaluation',
                    type=int, default=50)

parser.add_argument('--seed', help='manual seed to use, default is 123',
                    type=int, default=123)
parser.add_argument('--eval_every', help='how often to evaluate during training, never if None',
                    type=int, default=50)
parser.add_argument('--vis_every', help='how often to visualize during training, never if None',
                    type=int, default=50)
parser.add_argument('--save_every', help='how often to save during training, never if None',
                    type=int, default=100)
args = parser.parse_args()

if not torch.cuda.is_available() or args.device == 'cpu':
    args.device = torch.device('cpu')
else:
    if torch.cuda.device_count() == 1:
        # If you have CUDA_VISIBLE_DEVICES set, which you should,
        # then this will prevent leftover flag arguments from
        # messing with the device allocation.
        args.device = 'cuda:0'

    args.device = torch.device(args.device)

if args.eval_device is None:
    args.eval_device = args.device

if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

def main():
    # Load hyperparameters from json
    if not os.path.exists(args.conf):
        print('Config json not found!')
    with open(args.conf, 'r') as conf_json:
        hyperparams = json.load(conf_json)

    # Add hyperparams from arguments
    hyperparams['dynamic_edges'] = args.dynamic_edges
    hyperparams['edge_state_combine_method'] = args.edge_state_combine_method
    hyperparams['edge_influence_combine_method'] = args.edge_influence_combine_method
    hyperparams['edge_radius'] = args.edge_radius
    hyperparams['use_map_encoding'] = args.use_map_encoding
    hyperparams['edge_addition_filter'] = args.edge_addition_filter
    hyperparams['edge_removal_filter'] = args.edge_removal_filter
    hyperparams['batch_size'] = args.batch_size
    hyperparams['k_eval'] = args.k_eval
    hyperparams['offline_scene_graph'] = args.offline_scene_graph

    print('-----------------------')
    print('| TRAINING PARAMETERS |')
    print('-----------------------')
    print('| iterations: %d' % args.num_iters)
    print('| batch_size: %d' % args.batch_size)
    print('| batch_multiplier: %d' % args.batch_multiplier)
    print('| effective batch size: %d (= %d * %d)' % (args.batch_size * args.batch_multiplier, args.batch_size, args.batch_multiplier))
    print('| device: %s' % args.device)
    print('| eval_device: %s' % args.eval_device)
    print('| Offline Scene Graph Calculation: %s' % args.offline_scene_graph)
    print('| edge_radius: %s' % args.edge_radius)
    print('| EE state_combine_method: %s' % args.edge_state_combine_method)
    print('| EIE scheme: %s' % args.edge_influence_combine_method)
    print('| dynamic_edges: %s' % args.dynamic_edges)
    print('| robot node: %s' % args.incl_robot_node)
    print('| map encoding: %s' % args.use_map_encoding)
    print('| edge_addition_filter: %s' % args.edge_addition_filter)
    print('| edge_removal_filter: %s' % args.edge_removal_filter)
    print('| MHL: %s' % hyperparams['minimum_history_length'])
    print('| PH: %s' % hyperparams['prediction_horizon'])
    print('-----------------------')

    # Create the log and model directiory if they're not present.
    model_dir = os.path.join(args.log_dir, 'models_' + time.strftime('%d_%b_%Y_%H_%M_%S', time.localtime()))
    pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)

    # Save config to model directory
    with open(os.path.join(model_dir, 'config.json'), 'w') as conf_json:
        json.dump(hyperparams, conf_json)

    log_writer = SummaryWriter(log_dir=model_dir)

    train_scenes = []
    train_data_path = os.path.join(args.data_dir, args.train_data_dict)
    with open(train_data_path, 'rb') as f:
        train_env = pickle.load(f, encoding='latin1')
    train_scenes = train_env.scenes
    print('Loaded training data from %s' % (train_data_path,))

    eval_scenes = []
    if args.eval_every is not None:
        eval_data_path = os.path.join(args.data_dir, args.eval_data_dict)
        with open(eval_data_path, 'rb') as f:
            eval_env = pickle.load(f, encoding='latin1')
        eval_scenes = eval_env.scenes
        print('Loaded evaluation data from %s' % (eval_data_path, ))

    # Calculate Scene Graph
    if hyperparams['offline_scene_graph'] == 'yes':
        print(f"Offline calculating scene graphs")
        for i, scene in enumerate(train_scenes + eval_scenes):
            scene.calculate_scene_graph(hyperparams['edge_radius'],
                                        hyperparams['dims'],
                                        hyperparams['edge_addition_filter'],
                                        hyperparams['edge_removal_filter'])
            print(f"Created Scene Graph for Scene {i}")

    if args.incl_robot_node:
        #robot_node = stg_node.STGNode('0', 'Car')
        pass
    else:
        robot_node = None

    model_registrar = ModelRegistrar(model_dir, args.device)

    stg = SpatioTemporalGraphCVAEModel(robot_node,
                                       model_registrar,
                                       hyperparams,
                                       log_writer, args.device)
    stg.set_scene_graph(train_env)
    stg.set_annealing_params()
    print('Created training STG model.')

    eval_stg = None
    if args.eval_every is not None or args.vis_ervery is not None:
        eval_stg = SpatioTemporalGraphCVAEModel(robot_node,
                                                model_registrar,
                                                hyperparams,
                                                log_writer, args.device)
        eval_stg.set_scene_graph(eval_env)
        eval_stg.set_annealing_params() # TODO Check if necessary

    optimizer = optim.Adam(model_registrar.parameters(), lr=hyperparams['learning_rate'])
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=hyperparams['learning_decay_rate'])

    print_training_header(newline_start=True)
    for curr_iter in range(args.num_iters):
        # Necessary because we flip the weights contained between GPU and CPU sometimes.
        model_registrar.to(args.device)

        # Setting the current iterator value for internal logging.
        stg.set_curr_iter(curr_iter)

        # Stepping forward the learning rate scheduler and annealers.
        lr_scheduler.step()
        log_writer.add_scalar('train/learning_rate',
                              lr_scheduler.get_lr()[0],
                              curr_iter)
        stg.step_annealers()

        # Zeroing gradients for the upcoming iteration.
        optimizer.zero_grad()

        train_losses = list()
        for scene in train_scenes:
            for mb_num in range(args.batch_multiplier):
                # Obtaining the batch's training loss.
                timesteps = scene.sample_timesteps(hyperparams['batch_size'])

                # Compute the training loss.
                train_loss = stg.train_loss(scene, timesteps, max_nodes=hyperparams['batch_size'])
                if train_loss is not None:
                    train_loss = train_loss / (args.batch_multiplier * len(train_scenes))
                    train_losses.append(train_loss.item())

                    # Calculating gradients.
                    train_loss.backward()

        # Print training information. Also, no newline here. It's added in at a later line.
        iter_train_loss = sum(train_losses)
        print('{:9} | {:10} | '.format(curr_iter, '%.2f' % iter_train_loss),
              end='', flush=True)

        if len(train_losses) > 0:
            log_writer.add_histogram('train/minibatch_losses', np.asarray(train_losses), curr_iter)
            log_writer.add_scalar('train/loss', iter_train_loss, curr_iter)

        # Clipping gradients.
        if hyperparams['grad_clip'] is not None:
            nn.utils.clip_grad_value_(model_registrar.parameters(), hyperparams['grad_clip'])

        # Performing a gradient step.
        optimizer.step()

        del train_loss  # TODO Necessary?

        if args.vis_every is not None and (curr_iter + 1) % args.vis_every == 0:
            max_hl = hyperparams['maximum_history_length']
            ph = hyperparams['prediction_horizon']
            with torch.no_grad():
                # Predict random timestep to plot for train data set
                scene = np.random.choice(train_scenes)
                timestep = scene.sample_timesteps(1)
                predictions = stg.predict(scene,
                                          timestep,
                                          ph,
                                          num_samples_z=100,
                                          most_likely_z=False,
                                          all_z=False)

                # Plot predicted timestep for random scene
                fig, ax = plt.subplots(figsize=(5, 5))
                visualization.visualize_prediction(ax,
                                                   predictions,
                                                   scene.dt,
                                                   max_hl=max_hl,
                                                   ph=ph)
                log_writer.add_figure('train/prediction', fig, curr_iter)

                # Predict random timestep to plot for eval data set
                scene = np.random.choice(eval_scenes)
                timestep = scene.sample_timesteps(1)
                predictions = eval_stg.predict(scene,
                                               timestep,
                                               ph,
                                               num_samples_z=100,
                                               most_likely_z=False,
                                               all_z=False,
                                               max_nodes=4 * args.eval_batch_size)

                # Plot predicted timestep for random scene
                fig, ax = plt.subplots(figsize=(5, 5))
                visualization.visualize_prediction(ax,
                                                   predictions,
                                                   scene.dt,
                                                   max_hl=max_hl,
                                                   ph=ph)
                log_writer.add_figure('eval/prediction', fig, curr_iter)

                # Plot predicted timestep for random scene in map
                fig, ax = plt.subplots(figsize=(5, 5))
                visualization.visualize_prediction(ax,
                                                   predictions,
                                                   scene.dt,
                                                   max_hl=max_hl,
                                                   ph=ph,
                                                   map=scene.map)
                log_writer.add_figure('eval/prediction_map', fig, curr_iter)

                # Predict random timestep to plot for eval data set
                predictions = eval_stg.predict(scene,
                                               timestep,
                                               ph,
                                               num_samples_gmm=50,
                                               most_likely_z=False,
                                               all_z=True,
                                               max_nodes=4 * args.eval_batch_size)

                # Plot predicted timestep for random scene
                fig, ax = plt.subplots(figsize=(5, 5))
                visualization.visualize_prediction(ax,
                                                   predictions,
                                                   scene.dt,
                                                   max_hl=max_hl,
                                                   ph=ph)
                log_writer.add_figure('eval/prediction_all_z', fig, curr_iter)

        if args.eval_every is not None and (curr_iter + 1) % args.eval_every == 0:
            max_hl = hyperparams['maximum_history_length']
            ph = hyperparams['prediction_horizon']
            with torch.no_grad():
                # Predict batch timesteps for training dataset evaluation
                train_mse_batch_errors = np.array([])
                train_fde_batch_errors = np.array([])
                train_kde_nll = np.array([])
                train_obs_batch_viols = np.array([])
                for scene in train_scenes:
                    timesteps = scene.sample_timesteps(args.eval_batch_size)
                    predictions = stg.predict(scene,
                                              timesteps,
                                              ph,
                                              num_samples_z=200,
                                              min_future_timesteps=ph,
                                              max_nodes=4*args.eval_batch_size)

                    (train_mse_batch_errors_s,
                     train_fde_batch_errors_s,
                     train_kde_nll_s,
                     train_obs_batch_viols_s) = evaluation.compute_batch_statistics(predictions,
                                                                                  scene.dt,
                                                                                  max_hl=max_hl,
                                                                                  ph=ph,
                                                                                  map=scene.map)
                    train_mse_batch_errors = np.hstack((train_mse_batch_errors, train_mse_batch_errors_s))
                    train_fde_batch_errors = np.hstack((train_fde_batch_errors, train_fde_batch_errors_s))
                    train_kde_nll = np.hstack((train_kde_nll, train_kde_nll_s))
                    train_obs_batch_viols = np.hstack((train_obs_batch_viols, train_obs_batch_viols_s))

                log_writer.add_histogram('train/mse', train_mse_batch_errors, curr_iter)
                log_writer.add_histogram('train/fde', train_fde_batch_errors, curr_iter)
                log_writer.add_histogram('train/obs_viols', train_obs_batch_viols, curr_iter)
                log_writer.add_histogram('train/kde_nll', train_kde_nll, curr_iter)

                log_writer.add_scalar('train/obs_viols_mean', np.mean(train_obs_batch_viols), curr_iter)
                log_writer.add_scalar('train/mean_mse', np.mean(train_mse_batch_errors), curr_iter)
                log_writer.add_scalar('train/mean_fde', np.mean(train_fde_batch_errors), curr_iter)
                log_writer.add_scalar('train/median_mse', np.median(train_mse_batch_errors), curr_iter)
                log_writer.add_scalar('train/median_fde', np.median(train_fde_batch_errors), curr_iter)
                log_writer.add_scalar('train/mean_kde_nll', np.mean(train_kde_nll), curr_iter)
                log_writer.add_scalar('train/median_kde_nll', np.median(train_kde_nll), curr_iter)

                kde_fde_pd = {'dataset': ['ETH'] * train_kde_nll.shape[0],
                              'kde_nll': train_kde_nll.tolist()}
                kde_barplot_fig, ax = plt.subplots(figsize=(5, 5))
                visualization.visualization_utils.plot_barplots(ax, kde_fde_pd, 'dataset', 'kde_nll')
                log_writer.add_figure('train/kde_nll_barplot', kde_barplot_fig, curr_iter)

                mse_fde_pd = {'dataset': ['ETH'] * train_mse_batch_errors.shape[0],
                              'mse': train_mse_batch_errors.tolist(),
                              'fde': train_fde_batch_errors.tolist()}
                mse_boxplot_fig, ax = plt.subplots(figsize=(5, 5))
                visualization.visualization_utils.plot_boxplots(ax, mse_fde_pd, 'dataset', 'mse')
                fde_boxplot_fig, ax = plt.subplots(figsize=(5, 5))
                visualization.visualization_utils.plot_boxplots(ax, mse_fde_pd, 'dataset', 'fde')
                log_writer.add_figure('train/mse_boxplot', mse_boxplot_fig, curr_iter)
                log_writer.add_figure('train/fde_boxplot', fde_boxplot_fig, curr_iter)

                # Predict batch timesteps for evaluation dataset evaluation
                eval_mse_batch_errors = np.array([])
                eval_fde_batch_errors = np.array([])
                eval_kde_nll = np.array([])
                eval_obs_batch_viols = np.array([])
                for scene in eval_scenes:
                    timesteps = scene.sample_timesteps(args.eval_batch_size)

                    predictions = eval_stg.predict(scene,
                                                   timesteps,
                                                   ph,
                                                   num_samples_z=200,
                                                   min_future_timesteps=ph,
                                                   max_nodes=4 * args.eval_batch_size)

                    (eval_mse_batch_errors_s,
                     eval_fde_batch_errors_s,
                     eval_kde_nll_s,
                     eval_obs_batch_viols_s) = evaluation.compute_batch_statistics(predictions,
                                                                                 scene.dt,
                                                                                 max_hl=max_hl,
                                                                                 ph=ph,
                                                                                 map=scene.map)
                    eval_mse_batch_errors = np.hstack((eval_mse_batch_errors, eval_mse_batch_errors_s))
                    eval_fde_batch_errors = np.hstack((eval_fde_batch_errors, eval_fde_batch_errors_s))
                    eval_kde_nll = np.hstack((eval_kde_nll, eval_kde_nll_s))
                    eval_obs_batch_viols = np.hstack((eval_obs_batch_viols, eval_obs_batch_viols_s))

                log_writer.add_histogram('eval/mse', eval_mse_batch_errors, curr_iter)
                log_writer.add_histogram('eval/fde', eval_fde_batch_errors, curr_iter)
                log_writer.add_histogram('eval/obs_viols', eval_obs_batch_viols, curr_iter)
                log_writer.add_histogram('eval/kde_nll', eval_kde_nll, curr_iter)

                log_writer.add_scalar('eval/obs_viols_mean', np.mean(eval_obs_batch_viols), curr_iter)
                log_writer.add_scalar('eval/mean_mse', np.mean(eval_mse_batch_errors), curr_iter)
                log_writer.add_scalar('eval/mean_fde', np.mean(eval_fde_batch_errors), curr_iter)
                log_writer.add_scalar('eval/median_mse', np.median(eval_mse_batch_errors), curr_iter)
                log_writer.add_scalar('eval/median_fde', np.median(eval_fde_batch_errors), curr_iter)
                log_writer.add_scalar('eval/mean_kde_nll', np.mean(eval_kde_nll), curr_iter)
                log_writer.add_scalar('eval/median_kde_nll', np.median(eval_kde_nll), curr_iter)

                kde_fde_pd = {'dataset': ['ETH'] * eval_kde_nll.shape[0],
                              'kde_nll': eval_kde_nll.tolist()}
                kde_barplot_fig, ax = plt.subplots(figsize=(5, 5))
                visualization.visualization_utils.plot_barplots(ax, kde_fde_pd, 'dataset', 'kde_nll')
                log_writer.add_figure('eval/kde_nll_barplot', kde_barplot_fig, curr_iter)

                mse_fde_pd = {'dataset': ['ETH'] * eval_mse_batch_errors.shape[0],
                              'mse': eval_mse_batch_errors.tolist(),
                              'fde': eval_fde_batch_errors.tolist()}
                mse_boxplot_fig, ax = plt.subplots(figsize=(5, 5))
                visualization.visualization_utils.plot_boxplots(ax, mse_fde_pd, 'dataset', 'mse')
                fde_boxplot_fig, ax = plt.subplots(figsize=(5, 5))
                visualization.visualization_utils.plot_boxplots(ax, mse_fde_pd, 'dataset', 'fde')
                log_writer.add_figure('eval/mse_boxplot', mse_boxplot_fig, curr_iter)
                log_writer.add_figure('eval/fde_boxplot', fde_boxplot_fig, curr_iter)

                # Predict maximum likelihood batch timesteps for evaluation dataset evaluation
                eval_mse_batch_errors_ml = np.array([])
                eval_fde_batch_errors_ml = np.array([])
                for scene in eval_scenes:
                    timesteps = scene.sample_timesteps(args.eval_batch_size)

                    predictions = eval_stg.predict(scene,
                                                   timesteps,
                                                   ph,
                                                   num_samples_z=1,
                                                   min_future_timesteps=ph,
                                                   max_nodes=4 * args.eval_batch_size,
                                                   most_likely_z=True,
                                                   most_likely_gmm=True)

                    (eval_mse_batch_errors_ml_s,
                     eval_fde_batch_errors_ml_s,
                     _,
                     _) = evaluation.compute_batch_statistics(predictions,
                                                              scene.dt,
                                                              max_hl=max_hl,
                                                              ph=ph,
                                                              map=scene.map,
                                                              kde=False)
                    eval_mse_batch_errors_ml = np.hstack((eval_mse_batch_errors_ml, eval_mse_batch_errors_ml_s))
                    eval_fde_batch_errors_ml = np.hstack((eval_fde_batch_errors_ml, eval_fde_batch_errors_ml_s))

                log_writer.add_scalar('eval/mean_mse_ml', np.mean(eval_mse_batch_errors_ml), curr_iter)
                log_writer.add_scalar('eval/mean_fde_ml', np.mean(eval_fde_batch_errors_ml), curr_iter)
                log_writer.add_scalar('eval/median_mse_ml', np.median(eval_mse_batch_errors_ml), curr_iter)
                log_writer.add_scalar('eval/median_fde_ml', np.median(eval_fde_batch_errors_ml), curr_iter)

                eval_loss_q_is = np.array([])
                eval_loss_p = np.array([])
                eval_loss_exact = np.array([])
                eval_loss_sampled = np.array([])
                for scene in eval_scenes:
                    (eval_loss_q_is_s,
                     eval_loss_p_s,
                     eval_loss_exact_s,
                     eval_loss_sampled_s) = eval_stg.eval_loss(scene, timesteps)

                    eval_loss_q_is = np.hstack((eval_loss_q_is, eval_loss_q_is_s))
                    eval_loss_p = np.hstack((eval_loss_p, eval_loss_p_s))
                    eval_loss_exact = np.hstack((eval_loss_exact, eval_loss_exact_s))
                    eval_loss_sampled = np.hstack((eval_loss_sampled, eval_loss_sampled_s))

                log_writer.add_scalar('eval/loss/nll_q_is', np.mean(eval_loss_q_is), curr_iter)
                log_writer.add_scalar('eval/loss/nll_p', np.mean(eval_loss_p), curr_iter)
                log_writer.add_scalar('eval/loss/nll_exact', np.mean(eval_loss_exact), curr_iter)
                log_writer.add_scalar('eval/loss/nll_sampled', np.mean(eval_loss_sampled), curr_iter)

                print('{:15} | {:10} | {:14}'.format('%.2f' % np.mean(eval_loss_q_is).item(),
                                                     '%.2f' % np.mean(eval_loss_p).item(),
                                                     '%.2f' % np.mean(eval_loss_exact).item()),
                      end='', flush=True)

                # Freeing up memory.
                del eval_loss_q_is
                del eval_loss_p
                del eval_loss_exact

        else:
            print('{:15} | {:10} | {:14}'.format('', '', ''),
                  end='', flush=True)

        # Here's the newline that ends the current training information printing.
        print('')

        if args.save_every is not None and (curr_iter + 1) % args.save_every == 0:
            model_registrar.save_models(curr_iter)
            print_training_header()


def print_training_header(newline_start=False):
    if newline_start:
        print('')

    print('Iteration | Train Loss | Eval NLL Q (IS) | Eval NLL P | Eval NLL Exact')
    print('----------------------------------------------------------------------')


def memInUse():
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
    print('memory GB:', memoryUse)


if __name__ == '__main__':
    main()
