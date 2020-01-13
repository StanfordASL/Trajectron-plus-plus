import sys
import os
import numpy as np
import pandas as pd
import pickle

sys.path.append("../code")
from data import Environment, Scene, Node, Position, Velocity, Acceleration

desired_max_time = 100
pred_indices = [2, 3]
state_dim = 6
frame_diff = 10
desired_frame_diff = 1
dt = 0.4

standardization =  {}

standardization['position'] = {}
standardization['position']['x'] = {}
standardization['position']['y'] = {}
standardization['position']['x']['mean'] = 0.
standardization['position']['y']['mean'] = 0.

standardization['position']['x']['std'] = 1.
standardization['position']['y']['std'] = 1.

standardization['velocity'] = {}
standardization['velocity']['x'] = {}
standardization['velocity']['y'] = {}
standardization['velocity']['x']['mean'] = 0.
standardization['velocity']['y']['mean'] = 0.

standardization['velocity']['x']['std'] = 2.
standardization['velocity']['y']['std'] = 2.

standardization['acceleration'] = {}
standardization['acceleration']['x'] = {}
standardization['acceleration']['y'] = {}
standardization['acceleration']['x']['mean'] = 0.
standardization['acceleration']['y']['mean'] = 0.

standardization['acceleration']['x']['std'] = 1.
standardization['acceleration']['y']['std'] = 1.


for desired_source in ['eth', 'hotel', 'univ', 'zara1', 'zara2']:
    for data_class in ['train', 'val', 'test']:
        env = Environment(node_type_list=['PEDESTRIAN'], standardization=standardization)
        attention_radius = dict()
        attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN)] = 3.0
        env.attention_radius = attention_radius

        scenes = []
        data_dict_path = os.path.join('processed', '_'.join([desired_source, data_class]) + '.pkl')

        for subdir, dirs, files in os.walk(os.path.join('raw', desired_source, data_class)):
            for file in files:
                if file.endswith('.txt'):
                    input_data_dict = dict()
                    full_data_path = os.path.join(subdir, file)
                    full_data_path = os.path.join(subdir, file)
                    print('At', full_data_path)

                    data = pd.read_csv(full_data_path, sep='\t', index_col=False, header=None)
                    data.columns = ['frame_id', 'track_id', 'pos_x', 'pos_y']
                    data['frame_id'] = pd.to_numeric(data['frame_id'], downcast='integer')
                    data['track_id'] = pd.to_numeric(data['track_id'], downcast='integer')

                    data['frame_id'] = data['frame_id'] // 10

                    data['frame_id'] -= data['frame_id'].min()

                    data['node_type'] = 'PEDESTRIAN'
                    data['node_id'] = data['track_id'].astype(str)
                    data.sort_values('frame_id', inplace=True)

                    max_timesteps = data['frame_id'].max()

                    scene = Scene(timesteps=max_timesteps+1, dt=dt)

                    for node_id in pd.unique(data['node_id']):
                        node = Node(type=env.NodeType.PEDESTRIAN)

                        node_df = data[data['node_id'] == node_id]
                        assert np.all(np.diff(node_df['frame_id']) == 1)

                        node_values = node_df[['pos_x', 'pos_y']].values

                        if node_values.shape[0] < 2:
                            continue

                        new_first_idx = node_df['frame_id'].iloc[0]
                        node.first_timestep = new_first_idx

                        x = node_values[:, 0]
                        y = node_values[:, 1]

                        node.position = Position(x, y)
                        node.velocity = Velocity.from_position(node.position, scene.dt)
                        node.acceleration = Acceleration.from_velocity(node.velocity, scene.dt)

                        scene.nodes.append(node)

                    print(scene)
                    scenes.append(scene)
        print(f'Processed {len(scenes):.0f} scene for data class {data_class}')

        env.scenes = scenes

        if len(scenes) > 0:
            with open(data_dict_path, 'wb') as f:
                pickle.dump(env, f, protocol=pickle.HIGHEST_PROTOCOL)
