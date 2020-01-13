import sys
import os
import numpy as np
import pandas as pd
import pickle
import json
from tqdm import tqdm
from pyquaternion import Quaternion
from kalman_filter import LinearPointMass, NonlinearKinematicBicycle
from scipy.integrate import cumtrapz
from scipy.ndimage.morphology import binary_dilation, generate_binary_structure

nu_path = './nuscenes-devkit/python-sdk/'
#op_path = './pytorch-openpose/python/'
sys.path.append(nu_path)
sys.path.append("../../code")
#sys.path.append(op_path)
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from data import Environment, Scene, Node, BicycleNode, Position, Velocity, Acceleration, ActuatorAngle, Map, Scalar

scene_blacklist = [3, 12, 18, 19, 33, 35, 36, 41, 45, 50, 54, 55, 61, 120, 121, 123, 126, 132, 133, 134, 149,
                           154, 159, 196, 268, 278, 351, 365, 367, 368, 369, 372, 376, 377, 382, 385, 499, 515, 517,
                           945, 947, 952, 955, 962, 963, 968] + [969]


types = ['PEDESTRIAN',
         'BICYCLE',
         'VEHICLE']

standardization = {
    'PEDESTRIAN': {
        'position': {
            'x': {'mean': 0, 'std': 25},
            'y': {'mean': 0, 'std': 25}
        },
        'velocity': {
            'x': {'mean': 0, 'std': 2},
            'y': {'mean': 0, 'std': 2}
        },
        'acceleration': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1}
        },
        'heading': {
            'value': {'mean': 0, 'std': np.pi},
            'derivative': {'mean': 0, 'std': np.pi / 4}
        }
    },
    'BICYCLE': {
        'position': {
            'x': {'mean': 0, 'std': 50},
            'y': {'mean': 0, 'std': 50}
        },
        'velocity': {
            'x': {'mean': 0, 'std': 6},
            'y': {'mean': 0, 'std': 6},
            'm': {'mean': 0, 'std': 6}
        },
        'acceleration': {
            'x': {'mean': 0, 'std': 4},
            'y': {'mean': 0, 'std': 4},
            'm': {'mean': 0, 'std': 4}
        },
        'actuator_angle': {
            'steering_angle': {'mean': 0, 'std': np.pi/2}
        },
        'heading': {
            'value': {'mean': 0, 'std': np.pi},
            'derivative': {'mean': 0, 'std': np.pi / 4}
        }
    },
    'VEHICLE': {
        'position': {
            'x': {'mean': 0, 'std': 100},
            'y': {'mean': 0, 'std': 100}
        },
        'velocity': {
            'x': {'mean': 0, 'std': 20},
            'y': {'mean': 0, 'std': 20},
            'm': {'mean': 0, 'std': 20}
        },
        'acceleration': {
            'x': {'mean': 0, 'std': 4},
            'y': {'mean': 0, 'std': 4},
            'm': {'mean': 0, 'std': 4}
        },
        'actuator_angle': {
            'steering_angle': {'mean': 0, 'std': np.pi/2}
        },
        'heading': {
            'value': {'mean': 0, 'std': np.pi},
            'derivative': {'mean': 0, 'std': np.pi / 4}
        }
    }
}

def inverse_np_gradient(f, dx, F0=0.):
    N = f.shape[0]
    return F0 + np.hstack((np.zeros((N, 1)), cumtrapz(f, axis=1, dx=dx)))

def integrate_trajectory(v, x0, dt):
    xd_ = inverse_np_gradient(v[..., 0], dx=dt, F0=x0[0])
    yd_ = inverse_np_gradient(v[..., 1], dx=dt, F0=x0[1])
    integrated = np.stack([xd_, yd_], axis=2)
    return integrated

def integrate_heading_model(a, dh, h0, x0, v0, dt):
    h = inverse_np_gradient(dh, dx=dt, F0=h0)
    v_m = inverse_np_gradient(a, dx=dt, F0=v0)

    vx = np.cos(h) * v_m
    vy = np.sin(h) * v_m

    v = np.stack((vx, vy), axis=2)
    return integrate_trajectory(v, x0, dt)

if __name__ == "__main__":
    num_global_straight = 0
    num_global_curve = 0

    test = False
    if sys.argv[1] == 'mini':
        data_path = './raw_data/mini'
        nusc = NuScenes(version='v1.0-mini', dataroot=data_path, verbose=True)
        add = "_mini"
        train_scenes = nusc.scene[0:7]
        val_scenes = nusc.scene[7:]
        test_scenes = []
    elif sys.argv[1] == 'test':
        test = True
        data_path = './raw_data'
        nusc = NuScenes(version='v1.0-test', dataroot=data_path, verbose=True)
        train_scenes = []
        val_scenes = []
        test_scenes = nusc.scene
        with open(os.path.join('./raw_data/results_test_megvii.json'), 'r') as test_json:
            test_annotations = json.load(test_json)
    else:
        data_path = '/home/timsal/Documents/code/GenTrajectron_nuScenes_ssh/data/nuScenes/raw_data'
        nusc = NuScenes(version='v1.0-trainval', dataroot=data_path, verbose=True)
        add = ""
        train_scenes = nusc.scene[0:]
        val_scenes = nusc.scene[700:]
        test_scenes = []

    for data_class, nuscenes in [('train', train_scenes), ('val', val_scenes), ('test', test_scenes)]:
        print(f"Processing data class {data_class}")
        data_dict_path = os.path.join('../processed', '_'.join(['nuScenes', data_class])+ 'samp.pkl')
        env = Environment(node_type_list=types, standardization=standardization)
        attention_radius = dict()
        attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN)] = 3.0
        attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.VEHICLE)] = 20.0
        attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.BICYCLE)] = 10.0
        attention_radius[(env.NodeType.VEHICLE, env.NodeType.PEDESTRIAN)] = 20.0
        attention_radius[(env.NodeType.VEHICLE, env.NodeType.VEHICLE)] = 20.0
        attention_radius[(env.NodeType.VEHICLE, env.NodeType.BICYCLE)] = 20.0
        attention_radius[(env.NodeType.BICYCLE, env.NodeType.PEDESTRIAN)] = 10.0
        attention_radius[(env.NodeType.BICYCLE, env.NodeType.VEHICLE)] = 20.0
        attention_radius[(env.NodeType.BICYCLE, env.NodeType.BICYCLE)] = 10.0
        env.attention_radius = attention_radius
        scenes = []
        pbar = tqdm(nuscenes, ncols=100)
        for nuscene in pbar:
            scene_id = int(nuscene['name'].replace('scene-', ''))
            if scene_id in scene_blacklist: # Some scenes have bad localization
                continue

            if not (scene_id == 1002 or scene_id == 234):
                continue

            data = pd.DataFrame(columns=['frame_id',
                                         'type',
                                         'node_id',
                                         'robot',
                                         'x', 'y', 'z',
                                         'length',
                                         'width',
                                         'height',
                                         'heading',
                                         'orientation'])

            sample_token = nuscene['first_sample_token']
            sample = nusc.get('sample', sample_token)
            frame_id = 0
            while sample['next']:
                if not test:
                    annotation_tokens = sample['anns']
                else:
                    annotation_tokens = test_annotations['results'][sample['token']]
                for annotation_token in annotation_tokens:
                    if not test:
                        annotation = nusc.get('sample_annotation', annotation_token)
                        category = annotation['category_name']
                        if len(annotation['attribute_tokens']):
                            attribute = nusc.get('attribute', annotation['attribute_tokens'][0])['name']

                        if 'pedestrian' in category and not 'stroller' in category and not 'wheelchair' in category:
                            our_category = env.NodeType.PEDESTRIAN
                        elif ('vehicle.bicycle' in category) and 'with_rider' in attribute:
                            continue
                            our_category = env.NodeType.BICYCLE
                        elif 'vehicle' in category and 'bicycle' not in category and 'motorcycle' not in category and 'parked' not in attribute:
                            our_category = env.NodeType.VEHICLE
                        # elif ('vehicle.motorcycle' in category) and 'with_rider' in attribute:
                        # our_category = env.NodeType.VEHICLE
                        else:
                            continue
                    else:
                        annotation = annotation_token
                        category = annotation['tracking_name']
                        attribute = ""#annotation['attribute_name']

                        if 'pedestrian' in category :
                            our_category = env.NodeType.PEDESTRIAN
                        elif (('car' in category or 'bus' in category or 'construction_vehicle' in category) and 'parked' not in attribute):
                            our_category = env.NodeType.VEHICLE
                        # elif ('vehicle.motorcycle' in category) and 'with_rider' in attribute:
                        # our_category = env.NodeType.VEHICLE
                        else:
                            continue


                    data_point = pd.Series({'frame_id': frame_id,
                                            'type': our_category,
                                            'node_id': annotation['instance_token'] if not test else annotation['tracking_id'],
                                            'robot': False,
                                            'x': annotation['translation'][0],
                                            'y': annotation['translation'][1],
                                            'z': annotation['translation'][2],
                                            'length': annotation['size'][0],
                                            'width': annotation['size'][1],
                                            'height': annotation['size'][2],
                                            'heading': Quaternion(annotation['rotation']).yaw_pitch_roll[0],
                                            'orientation': None})
                    data = data.append(data_point, ignore_index=True)

                # Ego Vehicle
                our_category = env.NodeType.VEHICLE
                sample_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])
                annotation = nusc.get('ego_pose', sample_data['ego_pose_token'])
                data_point = pd.Series({'frame_id': frame_id,
                                        'type': our_category,
                                        'node_id': 'ego',
                                        'robot': True,
                                        'x': annotation['translation'][0],
                                        'y': annotation['translation'][1],
                                        'z': annotation['translation'][2],
                                        'length': 4,
                                        'width': 1.7,
                                        'height': 1.5,
                                        'heading': Quaternion(annotation['rotation']).yaw_pitch_roll[0],
                                        'orientation': None})
                data = data.append(data_point, ignore_index=True)

                sample = nusc.get('sample', sample['next'])
                frame_id += 1

            if len(data.index) == 0:
                continue

            data.sort_values('frame_id', inplace=True)
            max_timesteps = data['frame_id'].max()

            x_min = np.round(data['x'].min() - 50)
            x_max = np.round(data['x'].max() + 50)
            y_min = np.round(data['y'].min() - 50)
            y_max = np.round(data['y'].max() + 50)

            data['x'] = data['x'] - x_min
            data['y'] = data['y'] - y_min

            scene = Scene(timesteps=max_timesteps + 1, dt=0.5, name=str(scene_id))

            # Generate Maps
            map_name = nusc.get('log', nuscene['log_token'])['location']
            nusc_map = NuScenesMap(dataroot=data_path, map_name=map_name)

            type_map = dict()
            x_size = x_max - x_min
            y_size = y_max - y_min
            patch_box = (x_min + 0.5 * (x_max - x_min), y_min + 0.5 * (y_max - y_min), y_size, x_size)
            patch_angle = 0  # Default orientation where North is up
            canvas_size = (np.round(3 * y_size).astype(int), np.round(3 * x_size).astype(int))
            homography = np.array([[3., 0., 0.], [0., 3., 0.], [0., 0., 3.]])
            layer_names = ['lane', 'road_segment', 'drivable_area', 'road_divider', 'lane_divider', 'stop_line',
                           'ped_crossing', 'stop_line', 'ped_crossing', 'walkway']
            map_mask = (nusc_map.get_map_mask(patch_box, patch_angle, layer_names, canvas_size) * 255.0).astype(
                np.uint8)
            map_mask = np.swapaxes(map_mask, 1, 2) # x axis comes first
            # PEDESTRIANS
            map_mask_pedestrian = np.stack((map_mask[9], map_mask[8], np.max(map_mask[:3], axis=0)), axis=2)
            type_map['PEDESTRIAN'] = Map(data=map_mask_pedestrian, homography=homography,
                                         description=', '.join(layer_names))
            # Bicycles
            map_mask_bicycles = np.stack((map_mask[9], map_mask[8], np.max(map_mask[:3], axis=0)), axis=2)
            type_map['BICYCLE'] = Map(data=map_mask_bicycles, homography=homography, description=', '.join(layer_names))
            # VEHICLES
            map_mask_vehicle = np.stack((np.max(map_mask[:3], axis=0), map_mask[3], map_mask[4]), axis=2)
            type_map['VEHICLE'] = Map(data=map_mask_vehicle, homography=homography, description=', '.join(layer_names))

            map_mask_plot = np.stack(((np.max(map_mask[:3], axis=0) - (map_mask[3] + 0.5 * map_mask[4]).clip(
                max=255)).clip(min=0).astype(np.uint8), map_mask[8], map_mask[9]), axis=2)
            type_map['PLOT'] = Map(data=map_mask_plot, homography=homography, description=', '.join(layer_names))

            scene.map = type_map
            del map_mask
            del map_mask_pedestrian
            del map_mask_vehicle
            del map_mask_bicycles
            del map_mask_plot

            for node_id in pd.unique(data['node_id']):
                node_df = data[data['node_id'] == node_id]

                if node_df['x'].shape[0] < 2:
                    continue

                if not np.all(np.diff(node_df['frame_id']) == 1):
                    #print('Occlusion')
                    continue # TODO Make better

                node_values = node_df['x'].values
                if node_df.iloc[0]['type'] == env.NodeType.PEDESTRIAN:
                    node = Node(type=node_df.iloc[0]['type'])
                else:
                    node = BicycleNode(type=node_df.iloc[0]['type'])
                node.first_timestep = node_df['frame_id'].iloc[0]
                node.position = Position(node_df['x'].values, node_df['y'].values)
                node.velocity = Velocity.from_position(node.position, scene.dt)
                node.velocity.m = np.linalg.norm(np.vstack((node.velocity.x, node.velocity.y)), axis=0)
                node.acceleration = Acceleration.from_velocity(node.velocity, scene.dt)
                node.heading = Scalar(node_df['heading'].values)
                heading_t = node_df['heading'].values.copy()
                shifted_heading = np.zeros_like(node.heading.value)
                shifted_heading[0] = node.heading.value[0]
                for i in range(1, len(node.heading.value)):
                    if not (np.sign(node.heading.value[i]) == np.sign(node.heading.value[i - 1])) and np.abs(
                            node.heading.value[i]) > np.pi / 2:
                        shifted_heading[i] = shifted_heading[i - 1] + (
                                node.heading.value[i] - node.heading.value[i - 1]) - np.sign(
                            (node.heading.value[i] - node.heading.value[i - 1])) * 2 * np.pi
                    else:
                        shifted_heading[i] = shifted_heading[i - 1] + (
                                node.heading.value[i] - node.heading.value[i - 1])
                node.heading.value = shifted_heading
                node.length = node_df.iloc[0]['length']
                node.width = node_df.iloc[0]['width']

                if node_df.iloc[0]['robot'] == True:
                    node.is_robot = True

                if node_df.iloc[0]['type'] == env.NodeType.PEDESTRIAN:
                    filter_ped = LinearPointMass(dt=scene.dt)
                    for i in range(len(node.position.x)):
                        if i == 0:  # initalize KF
                            P_matrix = np.identity(4)
                        elif i < len(node.position.x):
                            # assign new est values
                            node.position.x[i] = x_vec_est_new[0][0]
                            node.velocity.x[i] = x_vec_est_new[1][0]
                            node.position.y[i] = x_vec_est_new[2][0]
                            node.velocity.y[i] = x_vec_est_new[3][0]

                        if i < len(node.position.x) - 1:  # no action on last data
                            # filtering
                            x_vec_est = np.array([[node.position.x[i]],
                                                  [node.velocity.x[i]],
                                                  [node.position.y[i]],
                                                  [node.velocity.y[i]]])
                            z_new = np.array([[node.position.x[i+1]],
                                              [node.position.y[i+1]]])
                            x_vec_est_new, P_matrix_new = filter_ped.predict_and_update(
                                x_vec_est=x_vec_est,
                                u_vec=np.array([[0.], [0.]]),
                                P_matrix=P_matrix,
                                z_new=z_new
                            )
                            P_matrix = P_matrix_new
                else:
                    filter_veh = NonlinearKinematicBicycle(lf=node.length*0.6, lr=node.length*0.4, dt=scene.dt)
                    for i in range(len(node.position.x)):
                        if i == 0:  # initalize KF
                            # initial P_matrix
                            P_matrix = np.identity(4)
                        elif i < len(node.position.x):
                            # assign new est values
                            node.position.x[i] = x_vec_est_new[0][0]
                            node.position.y[i] = x_vec_est_new[1][0]
                            node.heading.value[i] = x_vec_est_new[2][0]
                            node.velocity.m[i] = x_vec_est_new[3][0]

                        if i < len(node.position.x) - 1:  # no action on last data
                            # filtering
                            x_vec_est = np.array([[node.position.x[i]],
                                                  [node.position.y[i]],
                                                  [node.heading.value[i]],
                                                  [node.velocity.m[i]]])
                            z_new = np.array([[node.position.x[i+1]],
                                              [node.position.y[i+1]],
                                              [node.heading.value[i+1]],
                                              [node.velocity.m[i+1]]])
                            x_vec_est_new, P_matrix_new = filter_veh.predict_and_update(
                                x_vec_est=x_vec_est,
                                u_vec=np.array([[0.], [0.]]),
                                P_matrix=P_matrix,
                                z_new=z_new
                            )
                            P_matrix = P_matrix_new

                v_tmp = node.velocity.m
                node.velocity = Velocity.from_position(node.position, scene.dt)
                node.velocity.m = v_tmp
                #if (np.abs(np.linalg.norm(np.vstack((node.velocity.x, node.velocity.y)), axis=0) - v_tmp) > 0.4).any():
                #    print(np.abs(np.linalg.norm(np.vstack((node.velocity.x, node.velocity.y)), axis=0) - v_tmp))

                node.acceleration = Acceleration.from_velocity(node.velocity, scene.dt)
                node.acceleration.m = np.gradient(v_tmp, scene.dt)
                node.heading.derivative = np.gradient(node.heading.value, scene.dt)
                node.heading.value = (node.heading.value + np.pi) % (2.0 * np.pi) - np.pi

                if node_df.iloc[0]['type'] == env.NodeType.VEHICLE:
                    node_pos = np.stack((node.position.x, node.position.y), axis=1)
                    node_pos_map = scene.map[env.NodeType.VEHICLE.name].to_map_points(node_pos)
                    node_pos_int = np.round(node_pos_map).astype(int)
                    dilated_map = binary_dilation(scene.map[env.NodeType.VEHICLE.name].data[..., 0], generate_binary_structure(2, 2))
                    if np.sum((dilated_map[node_pos_int[:, 0], node_pos_int[:, 1]] == 0))/node_pos_int.shape[0] > 0.1:
                        del node
                        continue # Out of map



                if not node_df.iloc[0]['type'] == env.NodeType.PEDESTRIAN:
                    # Re Integrate:
                    i_pos = integrate_heading_model(np.array([node.acceleration.m[1:]]),
                                            np.array([node.heading.derivative[1:]]),
                                            node.heading.value[0],
                                            np.vstack((node.position.x[0], node.position.y[0])),
                                            node.velocity.m[0], 0.5)


                    #if (np.abs(node.heading.derivative) > np.pi/8).any():
                    #    print(np.abs(node.heading.derivative).max())
                scene.nodes.append(node)
                if node.is_robot is True:
                    scene.robot = node

            robot = False
            num_heading_changed = 0
            num_moving_vehicles = 0
            for node in scene.nodes:
                node.description = "straight"
                num_global_straight += 1
                if node.type == env.NodeType.VEHICLE:
                    if np.linalg.norm((node.position.x[0] - node.position.x[-1], node.position.y[0] - node.position.y[-1])) > 10:
                        num_moving_vehicles += 1
                    if np.abs(node.heading.value[0] - node.heading.value[-1]) > np.pi / 6:
                        if not np.sign(node.heading.value[0]) == np.sign(node.heading.value[-1]) and np.abs(node.heading.value[0] > 1/2 * np.pi):
                            if (node.heading.value[0] - node.heading.value[-1]) - np.sign((node.heading.value[0] - node.heading.value[-1])) * 2 * np.pi > np.pi / 6:
                                node.description = "curve"
                                num_global_curve += 1
                                num_global_straight -= 1
                                num_heading_changed += 1
                        else:
                            node.description = "curve"
                            num_global_curve += 1
                            num_global_straight -= 1
                            num_heading_changed += 1

                if node.is_robot:
                    robot = True

            if num_moving_vehicles > 0 and num_heading_changed / num_moving_vehicles > 0.4:
                scene.description = "curvy"
            else:
                scene.description = "straight"

            if robot: # If we dont have a ego vehicle there was bad localization
                pbar.set_description(str(scene))
                scenes.append(scene)

            del data

        env.scenes = scenes

        if len(scenes) > 0:
            with open(data_dict_path, 'wb') as f:
                pickle.dump(env, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(num_global_straight)
    print(num_global_curve)