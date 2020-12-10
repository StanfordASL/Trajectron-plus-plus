import copy
import numpy as np
from .scene_graph import TemporalSceneGraph, SceneGraph
from .node import MultiNode


class Scene(object):
    def __init__(self, timesteps, map=None, dt=1, name="", frequency_multiplier=1, aug_func=None,  non_aug_scene=None):
        self.map = map
        self.timesteps = timesteps
        self.dt = dt
        self.name = name

        self.nodes = []

        self.robot = None

        self.temporal_scene_graph = None

        self.frequency_multiplier = frequency_multiplier

        self.description = ""

        self.aug_func = aug_func
        self.non_aug_scene = non_aug_scene

    def add_robot_from_nodes(self, robot_type):
        scenes = [self]
        if hasattr(self, 'augmented'):
            scenes += self.augmented

        for scn in scenes:
            nodes_list = [node for node in scn.nodes if node.type == robot_type]
            non_overlapping_nodes = MultiNode.find_non_overlapping_nodes(nodes_list, min_timesteps=3)
            scn.robot = MultiNode(robot_type, 'ROBOT', non_overlapping_nodes, is_robot=True)

            for node in non_overlapping_nodes:
                scn.nodes.remove(node)
            scn.nodes.append(scn.robot)

    def get_clipped_input_dict(self, timestep, state):
        input_dict = dict()
        existing_nodes = self.get_nodes_clipped_at_time(timesteps=np.array([timestep]),
                                                        state=state)
        tr_scene = np.array([timestep, timestep])
        for node in existing_nodes:
            input_dict[node] = node.get(tr_scene, state[node.type])

        return input_dict

    def get_scene_graph(self,
                        timestep,
                        attention_radius=None,
                        edge_addition_filter=None,
                        edge_removal_filter=None) -> SceneGraph:
        """
        Returns the Scene Graph for a given timestep. If the Temporal Scene Graph was pre calculated,
        the temporal scene graph is sliced. Otherwise the scene graph is calculated on the spot.

        :param timestep: Timestep for which the scene graph is returned.
        :param attention_radius: Attention radius for each node type permutation. (Only online)
        :param edge_addition_filter: Filter for adding edges (Only online)
        :param edge_removal_filter:  Filter for removing edges (Only online)
        :return: Scene Graph for given timestep.
        """
        if self.temporal_scene_graph is None:
            timestep_range = np.array([timestep - len(edge_removal_filter), timestep])
            node_pos_dict = dict()
            present_nodes = self.present_nodes(np.array([timestep]))

            for node in present_nodes[timestep]:
                node_pos_dict[node] = np.squeeze(node.get(timestep_range, {'position': ['x', 'y']}))
            tsg = TemporalSceneGraph.create_from_temp_scene_dict(node_pos_dict,
                                                                 attention_radius,
                                                                 duration=(len(edge_removal_filter) + 1),
                                                                 edge_addition_filter=edge_addition_filter,
                                                                 edge_removal_filter=edge_removal_filter
                                                                 )

            return tsg.to_scene_graph(t=len(edge_removal_filter),
                                      t_hist=len(edge_removal_filter),
                                      t_fut=len(edge_addition_filter))
        else:
            return self.temporal_scene_graph.to_scene_graph(timestep,
                                                            len(edge_removal_filter),
                                                            len(edge_addition_filter))

    def calculate_scene_graph(self,
                              attention_radius,
                              edge_addition_filter=None,
                              edge_removal_filter=None) -> None:
        """
        Calculate the Temporal Scene Graph for the entire Scene.

        :param attention_radius: Attention radius for each node type permutation.
        :param edge_addition_filter: Filter for adding edges.
        :param edge_removal_filter: Filter for removing edges.
        :return: None
        """
        timestep_range = np.array([0, self.timesteps-1])
        node_pos_dict = dict()

        for node in self.nodes:
            if type(node) is MultiNode:
                node_pos_dict[node] = np.squeeze(node.get_all(timestep_range, {'position': ['x', 'y']}))
            else:
                node_pos_dict[node] = np.squeeze(node.get(timestep_range, {'position': ['x', 'y']}))

        self.temporal_scene_graph = TemporalSceneGraph.create_from_temp_scene_dict(node_pos_dict,
                                                                                   attention_radius,
                                                                                   duration=self.timesteps,
                                                                                   edge_addition_filter=edge_addition_filter,
                                                                                   edge_removal_filter=edge_removal_filter)

    def duration(self):
        """
        Calculates the duration of the scene.

        :return: Duration of the scene in s.
        """
        return self.timesteps * self.dt

    def present_nodes(self,
                      timesteps,
                      type=None,
                      min_history_timesteps=0,
                      min_future_timesteps=0,
                      return_robot=True) -> dict:
        """
        Finds all present nodes in the scene at a given timestemp

        :param timesteps: Timestep(s) for which all present nodes should be returned
        :param type: Node type which should be returned. If None all node types are returned.
        :param min_history_timesteps: Minimum history timesteps of a node to be returned.
        :param min_future_timesteps: Minimum future timesteps of a node to be returned.
        :param return_robot: Return a node if it is the robot.
        :return: Dictionary with timesteps as keys and list of nodes as value.
        """

        present_nodes = {}

        for node in self.nodes:
            if node.is_robot and not return_robot:
                continue
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

        return present_nodes

    def get_nodes_clipped_at_time(self, timesteps, state):
        clipped_nodes = list()

        existing_nodes = self.present_nodes(timesteps)
        all_nodes = set().union(*existing_nodes.values())
        if not all_nodes:
            return clipped_nodes

        tr_scene = np.array([timesteps.min(), timesteps.max()])
        data_header_memo = dict()
        for node in all_nodes:
            if isinstance(node, MultiNode):
                copied_node = copy.deepcopy(node.get_node_at_timesteps(tr_scene))
                copied_node.id = self.robot.id
            else:
                copied_node = copy.deepcopy(node)

            clipped_value = node.get(tr_scene, state[node.type])

            if node.type not in data_header_memo:
                data_header = list()
                for quantity, values in state[node.type].items():
                    for value in values:
                        data_header.append((quantity, value))

                data_header_memo[node.type] = data_header

            copied_node.overwrite_data(clipped_value, data_header_memo[node.type])
            copied_node.first_timestep = tr_scene[0]

            clipped_nodes.append(copied_node)

        return clipped_nodes

    def sample_timesteps(self, batch_size, min_future_timesteps=0) -> np.ndarray:
        """
        Sample a batch size of possible timesteps for the scene.

        :param batch_size: Number of timesteps to sample.
        :param min_future_timesteps: Minimum future timesteps in the scene for a timestep to be returned.
        :return: Numpy Array of sampled timesteps.
        """
        if batch_size > self.timesteps:
            batch_size = self.timesteps
        return np.random.choice(np.arange(0, self.timesteps-min_future_timesteps), size=batch_size, replace=False)

    def augment(self):
        if self.aug_func is not None:
            return self.aug_func(self)
        else:
            return self

    def get_node_by_id(self, id):
        for node in self.nodes:
            if node.id == id:
                return node

    def __repr__(self):
        return f"Scene: Duration: {self.duration()}s," \
               f" Nodes: {len(self.nodes)}," \
               f" Map: {'Yes' if self.map is not None else 'No'}."
