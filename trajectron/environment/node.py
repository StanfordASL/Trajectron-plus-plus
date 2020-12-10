import random
import numpy as np
import pandas as pd
from environment import DoubleHeaderNumpyArray
from ncls import NCLS


class Node(object):
    def __init__(self, node_type, node_id, data, length=None, width=None, height=None, first_timestep=0,
                 is_robot=False, description="", frequency_multiplier=1, non_aug_node=None):
        self.type = node_type
        self.id = node_id
        self.length = length
        self.width = width
        self.height = height
        self.first_timestep = first_timestep
        self.non_aug_node = non_aug_node

        if data is not None:
            if isinstance(data, pd.DataFrame):
                self.data = DoubleHeaderNumpyArray(data.values, list(data.columns))
            elif isinstance(data, DoubleHeaderNumpyArray):
                self.data = data
        else:
            self.data = None

        self.is_robot = is_robot
        self._last_timestep = None
        self.description = description
        self.frequency_multiplier = frequency_multiplier

        self.forward_in_time_on_next_override = False

    def __eq__(self, other):
        return ((isinstance(other, self.__class__)
                 or isinstance(self, other.__class__))
                and self.id == other.id
                and self.type == other.type)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.type, self.id))

    def __repr__(self):
        return '/'.join([self.type.name, self.id])

    def overwrite_data(self, data, header, forward_in_time_on_next_overwrite=False):
        """
        This function hard overwrites the data matrix. When using it you have to make sure that the columns
        in the new data matrix correspond to the old structure. As well as setting first_timestep.

        :param data: New data matrix
        :param forward_in_time_on_next_overwrite: On the !!NEXT!! call of overwrite_data first_timestep will be increased.
        :return:  None
        """
        if header is None:
            self.data.data = data
        else:
            self.data = DoubleHeaderNumpyArray(data, header)

        self._last_timestep = None
        if self.forward_in_time_on_next_override:
            self.first_timestep += 1
        self.forward_in_time_on_next_override = forward_in_time_on_next_overwrite

    def scene_ts_to_node_ts(self, scene_ts) -> (np.ndarray, int, int):
        """
        Transforms timestamp from scene into timeframe of node data.

        :param scene_ts: Scene timesteps
        :return: ts: Transformed timesteps, paddingl: Number of timesteps in scene range which are not available in
                node data before data is available. paddingu: Number of timesteps in scene range which are not
                available in node data after data is available.
        """
        paddingl = (self.first_timestep - scene_ts[0]).clip(0)
        paddingu = (scene_ts[1] - self.last_timestep).clip(0)
        ts = np.array(scene_ts).clip(min=self.first_timestep, max=self.last_timestep) - self.first_timestep
        return ts, paddingl, paddingu

    def history_points_at(self, ts) -> int:
        """
        Number of history points in trajectory. Timestep is exclusive.

        :param ts: Scene timestep where the number of history points are queried.
        :return: Number of history timesteps.
        """
        return ts - self.first_timestep

    def get(self, tr_scene, state, padding=np.nan) -> np.ndarray:
        """
        Returns a time range of multiple properties of the node.

        :param tr_scene: The timestep range (inklusive).
        :param state: The state description for which the properties are returned.
        :param padding: The value which should be used for padding if not enough information is available.
        :return: Array of node property values.
        """
        if tr_scene.size == 1:
            tr_scene = np.array([tr_scene[0], tr_scene[0]])
        length = tr_scene[1] - tr_scene[0] + 1  # tr is inclusive
        tr, paddingl, paddingu = self.scene_ts_to_node_ts(tr_scene)
        data_array = self.data[tr[0]:tr[1] + 1, state]
        padded_data_array = np.full((length, data_array.shape[1]), fill_value=padding)
        padded_data_array[paddingl:length - paddingu] = data_array
        return padded_data_array

    @property
    def timesteps(self) -> int:
        """
        Number of available timesteps for node.

        :return: Number of available timesteps.
        """
        return self.data.shape[0]

    @property
    def last_timestep(self) -> int:
        """
        Nodes last timestep in the Scene.

        :return: Nodes last timestep.
        """
        if self._last_timestep is None:
            self._last_timestep = self.first_timestep + self.timesteps - 1
        return self._last_timestep


class MultiNode(Node):
    def __init__(self, node_type, node_id, nodes_list, is_robot=False):
        super(MultiNode, self).__init__(node_type, node_id, data=None, is_robot=is_robot)
        self.nodes_list = nodes_list
        for node in self.nodes_list:
            node.is_robot = is_robot

        self.first_timestep = min(node.first_timestep for node in self.nodes_list)
        self._last_timestep = max(node.last_timestep for node in self.nodes_list)

        starts = np.array([node.first_timestep for node in self.nodes_list], dtype=np.int64)
        ends = np.array([node.last_timestep for node in self.nodes_list], dtype=np.int64)
        ids = np.arange(len(self.nodes_list), dtype=np.int64)
        self.interval_tree = NCLS(starts, ends, ids)

    @staticmethod
    def find_non_overlapping_nodes(nodes_list, min_timesteps=1) -> list:
        """
        Greedily finds a set of non-overlapping nodes in the provided scene.

        :return: A list of non-overlapping nodes.
        """
        non_overlapping_nodes = list()
        nodes = sorted(nodes_list, key=lambda n: n.last_timestep)
        current_time = 0
        for node in nodes:
            if node.first_timestep >= current_time and node.timesteps >= min_timesteps:
                # Include the node
                non_overlapping_nodes.append(node)
                current_time = node.last_timestep

        return non_overlapping_nodes

    def get_node_at_timesteps(self, scene_ts) -> Node:
        possible_node_ranges = list(self.interval_tree.find_overlap(scene_ts[0], scene_ts[1] + 1))
        if not possible_node_ranges:
            return Node(node_type=self.type,
                        node_id='EMPTY',
                        data=self.nodes_list[0].data * np.nan,
                        is_robot=self.is_robot)

        node_idx = random.choice(possible_node_ranges)[2]
        return self.nodes_list[node_idx]

    def scene_ts_to_node_ts(self, scene_ts) -> (Node, np.ndarray, int, int):
        """
        Transforms timestamp from scene into timeframe of node data.

        :param scene_ts: Scene timesteps
        :return: ts: Transformed timesteps, paddingl: Number of timesteps in scene range which are not available in
                node data before data is available. paddingu: Number of timesteps in scene range which are not
                available in node data after data is available.
        """
        possible_node_ranges = list(self.interval_tree.find_overlap(scene_ts[0], scene_ts[1] + 1))
        if not possible_node_ranges:
            return None, None, None, None

        node_idx = random.choice(possible_node_ranges)[2]
        node = self.nodes_list[node_idx]

        paddingl = (node.first_timestep - scene_ts[0]).clip(0)
        paddingu = (scene_ts[1] - node.last_timestep).clip(0)
        ts = np.array(scene_ts).clip(min=node.first_timestep, max=node.last_timestep) - node.first_timestep
        return node, ts, paddingl, paddingu

    def get(self, tr_scene, state, padding=np.nan) -> np.ndarray:
        if tr_scene.size == 1:
            tr_scene = np.array([tr_scene, tr_scene])
        length = tr_scene[1] - tr_scene[0] + 1  # tr is inclusive

        node, tr, paddingl, paddingu = self.scene_ts_to_node_ts(tr_scene)
        if node is None:
            state_length = sum([len(entity_dims) for entity_dims in state.values()])
            return np.full((length, state_length), fill_value=padding)

        data_array = node.data[tr[0]:tr[1] + 1, state]
        padded_data_array = np.full((length, data_array.shape[1]), fill_value=padding)
        padded_data_array[paddingl:length - paddingu] = data_array
        return padded_data_array

    def get_all(self, tr_scene, state, padding=np.nan) -> np.ndarray:
        # Assumption here is that the user is asking for all of the data in this MultiNode and to return it within a
        # full scene-sized output array.
        assert tr_scene.size == 2 and tr_scene[0] == 0 and self.last_timestep <= tr_scene[1]
        length = tr_scene[1] - tr_scene[0] + 1  # tr is inclusive
        state_length = sum([len(entity_dims) for entity_dims in state.values()])
        padded_data_array = np.full((length, state_length), fill_value=padding)
        for node in self.nodes_list:
            padded_data_array[node.first_timestep:node.last_timestep + 1] = node.data[:, state]

        return padded_data_array

    def history_points_at(self, ts) -> int:
        """
        Number of history points in trajectory. Timestep is exclusive.

        :param ts: Scene timestep where the number of history points are queried.
        :return: Number of history timesteps.
        """
        node_idx = next(self.interval_tree.find_overlap(ts, ts + 1))[2]
        node = self.nodes_list[node_idx]
        return ts - node.first_timestep

    @property
    def timesteps(self) -> int:
        """
        Number of available timesteps for node.

        :return: Number of available timesteps.
        """
        return self._last_timestep - self.first_timestep + 1
