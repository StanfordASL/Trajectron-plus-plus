import os
import time
import json
import torch
import pickle
import random
import pathlib
import evaluation
import numpy as np
import visualization as vis
from argument_parser import args
from model.online.online_trajectron import OnlineTrajectron
from model.model_registrar import ModelRegistrar
from data import Environment, Scene
import matplotlib.pyplot as plt

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


def create_online_env(env, hyperparams, scene_idx, init_timestep):
    test_scene = env.scenes[scene_idx]

    online_scene = Scene(timesteps=init_timestep + 1,
                         map=test_scene.map,
                         dt=test_scene.dt)
    online_scene.nodes = test_scene.get_nodes_clipped_at_time(
        timesteps=np.arange(init_timestep - hyperparams['maximum_history_length'],
                            init_timestep + 1),
        state=hyperparams['state'])
    online_scene.robot = test_scene.robot
    online_scene.calculate_scene_graph(attention_radius=env.attention_radius,
                                       edge_addition_filter=hyperparams['edge_addition_filter'],
                                       edge_removal_filter=hyperparams['edge_removal_filter'])

    return Environment(node_type_list=env.node_type_list,
                       standardization=env.standardization,
                       scenes=[online_scene],
                       attention_radius=env.attention_radius,
                       robot_type=env.robot_type)


def main():
    model_dir = os.path.join(args.log_dir, 'models_14_Jan_2020_00_24_21eth_no_rob')

    # Load hyperparameters from json
    config_file = os.path.join(model_dir, args.conf)
    if not os.path.exists(config_file):
        raise ValueError('Config json not found!')
    with open(config_file, 'r') as conf_json:
        hyperparams = json.load(conf_json)

    # Add hyperparams from arguments
    hyperparams['dynamic_edges'] = args.dynamic_edges
    hyperparams['edge_state_combine_method'] = args.edge_state_combine_method
    hyperparams['edge_influence_combine_method'] = args.edge_influence_combine_method
    hyperparams['edge_addition_filter'] = args.edge_addition_filter
    hyperparams['edge_removal_filter'] = args.edge_removal_filter
    hyperparams['batch_size'] = args.batch_size
    hyperparams['k_eval'] = args.k_eval
    hyperparams['offline_scene_graph'] = args.offline_scene_graph
    hyperparams['incl_robot_node'] = args.incl_robot_node
    hyperparams['scene_batch_size'] = args.scene_batch_size
    hyperparams['node_resample_train'] = args.node_resample_train
    hyperparams['node_resample_eval'] = args.node_resample_eval
    hyperparams['scene_resample_train'] = args.scene_resample_train
    hyperparams['scene_resample_eval'] = args.scene_resample_eval
    hyperparams['scene_resample_viz'] = args.scene_resample_viz
    hyperparams['edge_encoding'] = not args.no_edge_encoding

    output_save_dir = os.path.join(model_dir, 'pred_figs')
    pathlib.Path(output_save_dir).mkdir(parents=True, exist_ok=True)

    eval_data_path = os.path.join(args.data_dir, args.eval_data_dict)
    with open(eval_data_path, 'rb') as f:
        eval_env = pickle.load(f, encoding='latin1')

    if eval_env.robot_type is None and hyperparams['incl_robot_node']:
        eval_env.robot_type = eval_env.NodeType[0]  # TODO: Make more general, allow the user to specify?
        for scene in eval_env.scenes:
            scene.add_robot_from_nodes(eval_env.robot_type)

    print('Loaded evaluation data from %s' % (eval_data_path,))

    # Creating a dummy environment with a single scene that contains information about the world.
    # When using this code, feel free to use whichever scene index or initial timestep you wish.
    scene_idx = 0

    # You need to have at least acceleration, so you want 2 timesteps of prior data, e.g. [0, 1],
    # so that you can immediately start incremental inference from the 3rd timestep onwards.
    init_timestep = 1

    eval_scene = eval_env.scenes[scene_idx]
    online_env = create_online_env(eval_env, hyperparams, scene_idx, init_timestep)

    model_registrar = ModelRegistrar(model_dir, args.eval_device)
    model_registrar.load_models(iter_num=1999)

    trajectron = OnlineTrajectron(model_registrar,
                                  hyperparams,
                                  args.eval_device)

    # If you want to see what different robot futures do to the predictions, uncomment this line as well as
    # related "... += adjustment" lines below.
    # adjustment = np.stack([np.arange(13)/float(i*2.0) for i in range(6, 12)], axis=1)

    # Here's how you'd incrementally run the model, e.g. with streaming data.
    trajectron.set_environment(online_env, init_timestep)

    for timestep in range(init_timestep + 1, eval_scene.timesteps):
        pos_dict = eval_scene.get_clipped_pos_dict(timestep, hyperparams['state'])

        robot_present_and_future = None
        if eval_scene.robot is not None:
            robot_present_and_future = eval_scene.robot.get(np.array([timestep,
                                                                      timestep + hyperparams['prediction_horizon']]),
                                                            hyperparams['state'][eval_scene.robot.type],
                                                            padding=0.0)
            robot_present_and_future = np.stack([robot_present_and_future, robot_present_and_future], axis=0)
            # robot_present_and_future += adjustment

        start = time.time()
        preds = trajectron.incremental_forward(pos_dict,
                                               prediction_horizon=12,
                                               num_samples=25,
                                               robot_present_and_future=robot_present_and_future)
        end = time.time()
        print("t=%d: took %.2f s (= %.2f Hz) w/ %d nodes and %d edges" % (timestep, end - start,
                                                                          1. / (end - start), len(trajectron.nodes),
                                                                          trajectron.scene_graph.get_num_edges()))

        detailed_preds_dict = dict()
        for node in eval_scene.nodes:
            if node in preds:
                detailed_preds_dict[node] = preds[node]

        batch_stats = evaluation.compute_batch_statistics({timestep: detailed_preds_dict},
                                                          eval_scene.dt,
                                                          max_hl=hyperparams['maximum_history_length'],
                                                          ph=hyperparams['prediction_horizon'],
                                                          node_type_enum=online_env.NodeType,
                                                          prune_ph_to_future=True)

        evaluation.print_batch_errors([batch_stats], 'eval', timestep)

        fig, ax = plt.subplots()
        vis.visualize_prediction(ax,
                                 {timestep: preds},
                                 eval_scene.dt,
                                 hyperparams['maximum_history_length'],
                                 hyperparams['prediction_horizon'])

        if eval_scene.robot is not None:
            robot_for_plotting = eval_scene.robot.get(np.array([timestep,
                                                                timestep + hyperparams['prediction_horizon']]),
                                                      hyperparams['state'][eval_scene.robot.type])
            # robot_for_plotting += adjustment

            ax.plot(robot_for_plotting[1:, 1], robot_for_plotting[1:, 0],
                    color='r',
                    linewidth=1.0, alpha=1.0)

            # Current Node Position
            circle = plt.Circle((robot_for_plotting[0, 1],
                                 robot_for_plotting[0, 0]),
                                0.3,
                                facecolor='r',
                                edgecolor='k',
                                lw=0.5,
                                zorder=3)
            ax.add_artist(circle)

        fig.savefig(os.path.join(output_save_dir, f'pred_{timestep}.pdf'), dpi=300)
        plt.close(fig)


if __name__ == '__main__':
    main()
