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
from model.model_utils import cyclical_lr
from tensorboardX import SummaryWriter
#torch.autograd.set_detect_anomaly(True) # TODO Remove for speed


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
                    type=str, default='nuScenes_train.pkl')
parser.add_argument("--eval_data_dict", help="what file to load for evaluation data",
                    type=str, default='nuScenes_val.pkl')
parser.add_argument("--log_dir", help="what dir to save training information (i.e., saved models, logs, etc)",
                    type=str, default='../data/nuScenes/logs')
parser.add_argument("--log_tag", help="tag for the log folder",
                    type=str, default='')

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
    args.eval_device = 'cpu'

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
    hyperparams['incl_robot_node'] = args.incl_robot_node

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
    model_dir = os.path.join(args.log_dir, 'models_' + time.strftime('%d_%b_%Y_%H_%M_%S', time.localtime()) + args.log_tag)
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
        for i, scene in enumerate(train_scenes):
            scene.calculate_scene_graph(train_env.attention_radius,
                                        hyperparams['state'],
                                        hyperparams['edge_addition_filter'],
                                        hyperparams['edge_removal_filter'])
            print(f"Created Scene Graph for Scene {i}")

        for i, scene in enumerate(eval_scenes):
            scene.calculate_scene_graph(eval_env.attention_radius,
                                        hyperparams['state'],
                                        hyperparams['edge_addition_filter'],
                                        hyperparams['edge_removal_filter'])
            print(f"Created Scene Graph for Scene {i}")

    model_registrar = ModelRegistrar(model_dir, args.device)

    # We use pre trained weights for the map CNN
    if args.use_map_encoding:
        inf_encoder_registrar = os.path.join(args.log_dir, 'weight_trans/model_registrar-1499.pt')
        model_dict = torch.load(inf_encoder_registrar, map_location=args.device)

        for key in model_dict.keys():
            if 'map_encoder' in key:
                model_registrar.model_dict[key] = model_dict[key]
                assert model_registrar.get_model(key) is model_dict[key]

    stg = SpatioTemporalGraphCVAEModel(model_registrar,
                                       hyperparams,
                                       log_writer, args.device)
    stg.set_scene_graph(train_env)
    stg.set_annealing_params()
    print('Created training STG model.')

    eval_stg = None
    if args.eval_every is not None or args.vis_ervery is not None:
        eval_stg = SpatioTemporalGraphCVAEModel(model_registrar,
                                                hyperparams,
                                                log_writer, args.device)
        eval_stg.set_scene_graph(eval_env)
        eval_stg.set_annealing_params() # TODO Check if necessary
    if hyperparams['learning_rate_style'] == 'const':
        optimizer = optim.Adam(model_registrar.parameters(), lr=hyperparams['learning_rate'])
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.0)
    elif hyperparams['learning_rate_style'] == 'exp':
        optimizer = optim.Adam(model_registrar.parameters(), lr=hyperparams['learning_rate'])
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=hyperparams['learning_decay_rate'])
    elif hyperparams['learning_rate_style'] == 'triangle':
        optimizer = optim.Adam(model_registrar.parameters(), lr=1.0)
        clr = cyclical_lr(100, min_lr=hyperparams['min_learning_rate'], max_lr=hyperparams['learning_rate'], decay=hyperparams['learning_decay_rate'])
        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, [clr])

    print_training_header(newline_start=True)
    for curr_iter in range(args.num_iters):
        # Necessary because we flip the weights contained between GPU and CPU sometimes.
        model_registrar.to(args.device)

        # Setting the current iterator value for internal logging.
        stg.set_curr_iter(curr_iter)
        if args.vis_every is not None:
            eval_stg.set_curr_iter(curr_iter)

        # Stepping forward the learning rate scheduler and annealers.
        lr_scheduler.step()
        log_writer.add_scalar('train/learning_rate',
                              lr_scheduler.get_lr()[0],
                              curr_iter)
        stg.step_annealers()

        # Zeroing gradients for the upcoming iteration.
        optimizer.zero_grad()
        train_losses = dict()
        for node_type in train_env.NodeType:
            train_losses[node_type] = []
        for scene in np.random.choice(train_scenes, 10):
            for mb_num in range(args.batch_multiplier):
                # Obtaining the batch's training loss.
                timesteps = scene.sample_timesteps(hyperparams['batch_size'])

                # Compute the training loss.
                train_loss_by_type = stg.train_loss(scene, timesteps, max_nodes=hyperparams['batch_size'])
                for node_type, train_loss in train_loss_by_type.items():
                    if train_loss is not None:
                        train_loss = train_loss / (args.batch_multiplier * 10)
                        train_losses[node_type].append(train_loss.item())

                        # Calculating gradients.
                        train_loss.backward()

        # Print training information. Also, no newline here. It's added in at a later line.
        print('{:9} | '.format(curr_iter), end='', flush=True)
        for node_type in train_env.NodeType:
            print('{}:{:10} | '.format(node_type.name[0], '%.2f' % sum(train_losses[node_type])), end='', flush=True)

        for node_type in train_env.NodeType:
            if len(train_losses[node_type]) > 0:
                log_writer.add_histogram(f"{node_type.name}/train/minibatch_losses", np.asarray(train_losses[node_type]), curr_iter)
                log_writer.add_scalar(f"{node_type.name}/train/loss", sum(train_losses[node_type]), curr_iter)

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
                timestep = scene.sample_timesteps(1, min_future_timesteps=ph)
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
                ax.set_title(f"{scene.name}-t: {timestep}")
                log_writer.add_figure('train/prediction', fig, curr_iter)

                # Predict random timestep to plot for eval data set
                scene = np.random.choice(eval_scenes)
                timestep = scene.sample_timesteps(1, min_future_timesteps=ph)
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
                ax.set_title(f"{scene.name}-t: {timestep}")
                log_writer.add_figure('eval/prediction', fig, curr_iter)

                # Plot predicted timestep for random scene in map
                fig, ax = plt.subplots(figsize=(15, 15))
                visualization.visualize_prediction(ax,
                                                   predictions,
                                                   scene.dt,
                                                   max_hl=max_hl,
                                                   ph=ph,
                                                   map=scene.map['PLOT'])
                ax.set_title(f"{scene.name}-t: {timestep}")
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
                ax.set_title(f"{scene.name}-t: {timestep}")
                log_writer.add_figure('eval/prediction_all_z', fig, curr_iter)

        if args.eval_every is not None and (curr_iter + 1) % args.eval_every == 0:
            max_hl = hyperparams['maximum_history_length']
            ph = hyperparams['prediction_horizon']
            with torch.no_grad():
                # Predict batch timesteps for training dataset evaluation
                train_batch_errors = []
                max_scenes = np.min([len(train_scenes), 5])
                for scene in np.random.choice(train_scenes, max_scenes):
                    timesteps = scene.sample_timesteps(args.eval_batch_size)
                    predictions = stg.predict(scene,
                                              timesteps,
                                              ph,
                                              num_samples_z=100,
                                              min_future_timesteps=ph,
                                              max_nodes=4*args.eval_batch_size)

                    train_batch_errors.append(evaluation.compute_batch_statistics(predictions,
                                                                                  scene.dt,
                                                                                  max_hl=max_hl,
                                                                                  ph=ph,
                                                                                  node_type_enum=train_env.NodeType,
                                                                                  map=scene.map))

                evaluation.log_batch_errors(train_batch_errors,
                                            log_writer,
                                            'train',
                                            curr_iter,
                                            bar_plot=['kde'],
                                            box_plot=['ade', 'fde'])

                # Predict batch timesteps for evaluation dataset evaluation
                eval_batch_errors = []
                for scene in eval_scenes:
                    timesteps = scene.sample_timesteps(args.eval_batch_size)

                    predictions = eval_stg.predict(scene,
                                                   timesteps,
                                                   ph,
                                                   num_samples_z=100,
                                                   min_future_timesteps=ph,
                                                   max_nodes=4 * args.eval_batch_size)

                    eval_batch_errors.append(evaluation.compute_batch_statistics(predictions,
                                                                                 scene.dt,
                                                                                 max_hl=max_hl,
                                                                                 ph=ph,
                                                                                 node_type_enum=eval_env.NodeType,
                                                                                 map=scene.map))

                evaluation.log_batch_errors(eval_batch_errors,
                                            log_writer,
                                            'eval',
                                            curr_iter,
                                            bar_plot=['kde'],
                                            box_plot=['ade', 'fde'])


                # Predict maximum likelihood batch timesteps for evaluation dataset evaluation
                eval_batch_errors_ml = []
                for scene in eval_scenes:
                    timesteps = scene.sample_timesteps(scene.timesteps)

                    predictions = eval_stg.predict(scene,
                                                   timesteps,
                                                   ph,
                                                   num_samples_z=1,
                                                   min_future_timesteps=ph,
                                                   most_likely_z=True,
                                                   most_likely_gmm=True)

                    eval_batch_errors_ml.append(evaluation.compute_batch_statistics(predictions,
                                                              scene.dt,
                                                              max_hl=max_hl,
                                                              ph=ph,
                                                              map=scene.map,
                                                              node_type_enum=eval_env.NodeType,
                                                              kde=False))

                evaluation.log_batch_errors(eval_batch_errors_ml,
                                            log_writer,
                                            'eval/ml',
                                            curr_iter)

                eval_loss = []
                max_scenes = np.min([len(eval_scenes), 25])
                for scene in np.random.choice(eval_scenes, max_scenes):
                    eval_loss.append(eval_stg.eval_loss(scene, timesteps))

                evaluation.log_batch_errors(eval_loss,
                                            log_writer,
                                            'eval/loss',
                                            curr_iter)


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
