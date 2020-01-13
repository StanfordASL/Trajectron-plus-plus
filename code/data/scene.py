import numpy as np
from .scene_graph import TemporalSceneGraph


class Scene(object):
    def __init__(self, map=None, timesteps=0, dt=1):
        self.map = map
        self.timesteps = timesteps
        self.dt = dt

        self.nodes = []

        self.robot = None

        self.temporal_scene_graph = None

    def get_scene_graph(self, timestep, dims, edge_radius=None, edge_addition_filter=None, edge_removal_filter=None):
        if self.temporal_scene_graph is None:
            timestep_range = np.array([timestep - len(edge_addition_filter), timestep + len(edge_removal_filter)])
            node_pos_dict = dict()
            present_nodes = self.present_nodes(np.array([timestep]))

            for node in present_nodes[timestep]:
                node_pos_dict[node] = np.squeeze(node.get(timestep_range, 'position', dims=dims))
            tsg = TemporalSceneGraph.create_from_temp_scene_dict(node_pos_dict,
                                                                 edge_radius,
                                                                 duration=(len(edge_addition_filter) +
                                                                       len(edge_removal_filter) + 1),
                                                                 edge_addition_filter=edge_addition_filter,
                                                                 edge_removal_filter=edge_removal_filter
                                                                 )

            return tsg.to_scene_graph(t=len(edge_addition_filter),
                                      t_hist=len(edge_addition_filter),
                                      t_fut=len(edge_removal_filter))
        else:
            return self.temporal_scene_graph.to_scene_graph(timestep,
                                                            len(edge_addition_filter),
                                                            len(edge_removal_filter))

    def calculate_scene_graph(self, edge_radius, dims, edge_addition_filter=None, edge_removal_filter=None):
        timestep_range = np.array([0, self.timesteps-1])
        node_pos_dict = dict()

        for node in self.nodes:
            node_pos_dict[node] = np.squeeze(node.get(timestep_range, 'position', dims=dims))

        self.temporal_scene_graph = TemporalSceneGraph.create_from_temp_scene_dict(node_pos_dict,
                                                                                   edge_radius,
                                                                                   duration=self.timesteps,
                                                                                   edge_addition_filter=edge_addition_filter,
                                                                                   edge_removal_filter=edge_removal_filter)

    def length(self):
        return self.timesteps * self.dt

    def present_nodes(self, timesteps, type=None, min_history_timesteps=0, min_future_timesteps=0, max_nodes=None):
        present_nodes = {}

        picked_nodes = 0

        rand_idx = np.random.choice(len(self.nodes), len(self.nodes), replace=False)

        for i in rand_idx:
            node = self.nodes[i]
            if type is None or node.type == type:
                lower_bound = timesteps - min_history_timesteps
                upper_bound = timesteps + min_future_timesteps
                mask = (node.first_timestep <= lower_bound) & (upper_bound <= node.last_timestep)
                if mask.any():
                    timestep_indices_present = np.nonzero(mask)[0]
                    for timestep_index_present in timestep_indices_present:
                        if timesteps[timestep_index_present] in present_nodes.keys():
                            present_nodes[timesteps[timestep_index_present]].append(node)
                        else:
                            present_nodes[timesteps[timestep_index_present]] = [node]
                        picked_nodes += 1
                        if max_nodes is not None and picked_nodes >= max_nodes:
                            break

            if max_nodes is not None and picked_nodes >= max_nodes:
                break

        return present_nodes

    def sample_timesteps(self, batch_size):
        if batch_size > self.timesteps:
            batch_size = self.timesteps
        return np.random.choice(np.arange(0, self.timesteps), size=batch_size, replace=False)

    def __repr__(self):
        return f"Scene: Duration: {self.length()}s," \
               f" Nodes: {len(self.nodes)}," \
               f" Map: {'Yes' if self.map is not None else 'No'}."
