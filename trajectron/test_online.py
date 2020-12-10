import os
import time
import json
import torch
import dill
import random
import pathlib
import evaluation
import numpy as np
import visualization as vis
from argument_parser import args
from model.online.online_trajectron import OnlineTrajectron
from model.model_registrar import ModelRegistrar
from environment import Environment, Scene
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


def get_maps_for_input(input_dict, scene, hyperparams):
    scene_maps = list()
    scene_pts = list()
    heading_angles = list()
    patch_sizes = list()
    nodes_with_maps = list()
    for node in input_dict:
        if node.type in hyperparams['map_encoder']:
            x = input_dict[node]
            me_hyp = hyperparams['map_encoder'][node.type]
            if 'heading_state_index' in me_hyp:
                heading_state_index = me_hyp['heading_state_index']
                # We have to rotate the map in the opposit direction of the agent to match them
                if type(heading_state_index) is list:  # infer from velocity or heading vector
                    heading_angle = -np.arctan2(x[-1, heading_state_index[1]],
                                                x[-1, heading_state_index[0]]) * 180 / np.pi
                else:
                    heading_angle = -x[-1, heading_state_index] * 180 / np.pi
            else:
                heading_angle = None

            scene_map = scene.map[node.type]
            map_point = x[-1, :2]

            patch_size = hyperparams['map_encoder'][node.type]['patch_size']

            scene_maps.append(scene_map)
            scene_pts.append(map_point)
            heading_angles.append(heading_angle)
            patch_sizes.append(patch_size)
            nodes_with_maps.append(node)

    if heading_angles[0] is None:
        heading_angles = None
    else:
        heading_angles = torch.Tensor(heading_angles)

    maps = scene_maps[0].get_cropped_maps_from_scene_map_batch(scene_maps,
                                                               scene_pts=torch.Tensor(scene_pts),
                                                               patch_size=patch_sizes[0],
                                                               rotation=heading_angles)

    maps_dict = {node: maps[[i]] for i, node in enumerate(nodes_with_maps)}
    return maps_dict


def main():
    # Choose one of the model directory names under the experiment/*/models folders.
    # Possibilities are 'vel_ee', 'int_ee', 'int_ee_me', or 'robot'
    model_dir = os.path.join(args.log_dir, 'int_ee')

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
    hyperparams['edge_encoding'] = not args.no_edge_encoding
    hyperparams['use_map_encoding'] = args.map_encoding

    output_save_dir = os.path.join(model_dir, 'pred_figs')
    pathlib.Path(output_save_dir).mkdir(parents=True, exist_ok=True)

    eval_data_path = os.path.join(args.data_dir, args.eval_data_dict)
    with open(eval_data_path, 'rb') as f:
        eval_env = dill.load(f, encoding='latin1')

    if eval_env.robot_type is None and hyperparams['incl_robot_node']:
        eval_env.robot_type = eval_env.NodeType[0]  # TODO: Make more general, allow the user to specify?
        for scene in eval_env.scenes:
            scene.add_robot_from_nodes(eval_env.robot_type)

    print('Loaded data from %s' % (eval_data_path,))

    # Creating a dummy environment with a single scene that contains information about the world.
    # When using this code, feel free to use whichever scene index or initial timestep you wish.
    scene_idx = 0

    # You need to have at least acceleration, so you want 2 timesteps of prior data, e.g. [0, 1],
    # so that you can immediately start incremental inference from the 3rd timestep onwards.
    init_timestep = 1

    eval_scene = eval_env.scenes[scene_idx]
    online_env = create_online_env(eval_env, hyperparams, scene_idx, init_timestep)

    model_registrar = ModelRegistrar(model_dir, args.eval_device)
    model_registrar.load_models(iter_num=12)

    trajectron = OnlineTrajectron(model_registrar,
                                  hyperparams,
                                  args.eval_device)

    # If you want to see what different robot futures do to the predictions, uncomment this line as well as
    # related "... += adjustment" lines below.
    # adjustment = np.stack([np.arange(13)/float(i*2.0) for i in range(6, 12)], axis=1)

    # Here's how you'd incrementally run the model, e.g. with streaming data.
    trajectron.set_environment(online_env, init_timestep)

    for timestep in range(init_timestep + 1, eval_scene.timesteps):
        input_dict = eval_scene.get_clipped_input_dict(timestep, hyperparams['state'])

        maps = None
        if hyperparams['use_map_encoding']:
            maps = get_maps_for_input(input_dict, eval_scene, hyperparams)

        robot_present_and_future = None
        if eval_scene.robot is not None and hyperparams['incl_robot_node']:
            robot_present_and_future = eval_scene.robot.get(np.array([timestep,
                                                                      timestep + hyperparams['prediction_horizon']]),
                                                            hyperparams['state'][eval_scene.robot.type],
                                                            padding=0.0)
            robot_present_and_future = np.stack([robot_present_and_future, robot_present_and_future], axis=0)
            # robot_present_and_future += adjustment

        start = time.time()
        dists, preds = trajectron.incremental_forward(input_dict,
                                                      maps,
                                                      prediction_horizon=6,
                                                      num_samples=1,
                                                      robot_present_and_future=robot_present_and_future,
                                                      full_dist=True)
        end = time.time()
        print("t=%d: took %.2f s (= %.2f Hz) w/ %d nodes and %d edges" % (timestep, end - start,
                                                                          1. / (end - start), len(trajectron.nodes),
                                                                          trajectron.scene_graph.get_num_edges()))

        detailed_preds_dict = dict()
        for node in eval_scene.nodes:
            if node in preds:
                detailed_preds_dict[node] = preds[node]

        fig, ax = plt.subplots()
        vis.visualize_distribution(ax,
                                   dists)
        vis.visualize_prediction(ax,
                                 {timestep: preds},
                                 eval_scene.dt,
                                 hyperparams['maximum_history_length'],
                                 hyperparams['prediction_horizon'])

        if eval_scene.robot is not None and hyperparams['incl_robot_node']:
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
