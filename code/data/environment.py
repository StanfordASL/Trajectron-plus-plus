import numpy as np
from enum import Enum
from itertools import product


class Environment(object):
    def __init__(self, node_type_list, standardization, scenes=None, attention_radius=None):
        self.scenes = scenes
        self.node_type_list = node_type_list
        self.attention_radius = attention_radius
        self.NodeType = Enum('NodeType', node_type_list)

        self.standardization = standardization

    def get_edge_types(self):
        return [e for e in product([node_type for node_type in self.NodeType], repeat=2)]

    def edge_type_str(self, edge_type):
        return edge_type[0].name + '-' + edge_type[1].name

    def get_standardize_params(self, state, node_type):
        standardize_mean_list = list()
        standardize_std_list = list()
        for entity, dims in state.items():
            for dim in dims:
                standardize_mean_list.append(self.standardization[node_type.name][entity][dim]['mean'])
                standardize_std_list.append(self.standardization[node_type.name][entity][dim]['std'])
        standardize_mean = np.stack(standardize_mean_list)
        standardize_std = np.stack(standardize_std_list)

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

    # These two functions have to be implemented as pickle can not handle dynamic enums
    def __getstate__(self):
        for scene in self.scenes:
            for node in scene.nodes:
                node.type = node.type.name
        attention_radius_no_enum = dict()
        for key, value in self.attention_radius.items():
            attention_radius_no_enum[(key[0].name, key[1].name)] = value
        self.attention_radius = attention_radius_no_enum
        self.NodeType = None
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.NodeType = Enum('NodeType', self.node_type_list)
        for scene in self.scenes:
            for node in scene.nodes:
                node.type = getattr(self.NodeType, node.type)
        attention_radius_enum = dict()
        for key, value in self.attention_radius.items():
            attention_radius_enum[(getattr(self.NodeType, key[0]), getattr(self.NodeType, key[1]))] = value
        self.attention_radius = attention_radius_enum

