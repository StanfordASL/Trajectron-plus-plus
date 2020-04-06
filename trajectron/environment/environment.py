import orjson
import numpy as np
from itertools import product
from .node_type import NodeTypeEnum


class Environment(object):
    def __init__(self, node_type_list, standardization, scenes=None, attention_radius=None, robot_type=None):
        self.scenes = scenes
        self.node_type_list = node_type_list
        self.attention_radius = attention_radius
        self.NodeType = NodeTypeEnum(node_type_list)
        self.robot_type = robot_type

        self.standardization = standardization
        self.standardize_param_memo = dict()

        self._scenes_resample_prop = None

    def get_edge_types(self):
        return list(product(self.NodeType, repeat=2))

    def get_standardize_params(self, state, node_type):
        memo_key = (orjson.dumps(state), node_type)
        if memo_key in self.standardize_param_memo:
            return self.standardize_param_memo[memo_key]

        standardize_mean_list = list()
        standardize_std_list = list()
        for entity, dims in state.items():
            for dim in dims:
                standardize_mean_list.append(self.standardization[node_type][entity][dim]['mean'])
                standardize_std_list.append(self.standardization[node_type][entity][dim]['std'])
        standardize_mean = np.stack(standardize_mean_list)
        standardize_std = np.stack(standardize_std_list)

        self.standardize_param_memo[memo_key] = (standardize_mean, standardize_std)
        return standardize_mean, standardize_std

    def standardize(self, array, state, node_type, mean=None, std=None):
        if mean is None and std is None:
            mean, std = self.get_standardize_params(state, node_type)
        elif mean is None and std is not None:
            mean, _ = self.get_standardize_params(state, node_type)
        elif mean is not None and std is None:
            _, std = self.get_standardize_params(state, node_type)
        return np.where(np.isnan(array), np.array(np.nan), (array - mean) / std)

    def unstandardize(self, array, state, node_type, mean=None, std=None):
        if mean is None and std is None:
            mean, std = self.get_standardize_params(state, node_type)
        elif mean is None and std is not None:
            mean, _ = self.get_standardize_params(state, node_type)
        elif mean is not None and std is None:
            _, std = self.get_standardize_params(state, node_type)
        return array * std + mean

    @property
    def scenes_resample_prop(self):
        if self._scenes_resample_prop is None:
            self._scenes_resample_prop = np.array([scene.resample_prob for scene in self.scenes])
            self._scenes_resample_prop = self._scenes_resample_prop / np.sum(self._scenes_resample_prop)
        return self._scenes_resample_prop

