import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict, Counter
from model.components import *
from model.model_utils import *
from model.dataset import get_relative_robot_traj
import model.dynamics as dynamic_module
from model.mgcvae import MultimodalGenerativeCVAE
from environment.scene_graph import DirectedEdge
from environment.node_type import NodeType


class OnlineMultimodalGenerativeCVAE(MultimodalGenerativeCVAE):
    def __init__(self,
                 env,
                 node,
                 model_registrar,
                 hyperparams,
                 device):
        self.hyperparams = hyperparams
        self.node = node
        self.node_type = self.node.type

        if len(env.scenes) != 1:
            raise ValueError("Passed in Environment has number of scenes != 1")
        self.robot = env.scenes[0].robot
        self.model_registrar = model_registrar
        self.device = device

        self.node_modules = dict()
        self.env = env
        self.scene_graph = None

        self.state = self.hyperparams['state']
        self.pred_state = self.hyperparams['pred_state'][self.node.type]
        self.state_length = int(np.sum([len(entity_dims) for entity_dims in self.state[self.node.type].values()]))
        if self.hyperparams['incl_robot_node']:
            self.robot_state_length = int(
                np.sum([len(entity_dims) for entity_dims in self.state[self.robot.type].values()]))
        self.pred_state_length = int(np.sum([len(entity_dims) for entity_dims in self.pred_state.values()]))

        self.curr_hidden_states = dict()
        self.edge_types = Counter()

        self.create_graphical_model()

        dynamic_class = getattr(dynamic_module, self.hyperparams['dynamic'][self.node_type]['name'])
        dyn_limits = hyperparams['dynamic'][self.node_type]['limits']
        self.dynamic = dynamic_class(self.env.scenes[0].dt, dyn_limits, device,
                                     self.model_registrar, self.x_size, self.node_type)

    def create_graphical_model(self):
        """
        Creates or queries all trainable components.

        :return: None
        """
        self.clear_submodules()

        ############################
        #   Everything but Edges   #
        ############################
        self.create_node_models()

        for name, module in self.node_modules.items():
            module.to(self.device)

    def update_graph(self, new_scene_graph, new_neighbors, removed_neighbors):
        self.scene_graph = new_scene_graph

        if self.node in new_neighbors:
            for edge_type, new_neighbor_nodes in new_neighbors[self.node].items():
                self.add_edge_model(edge_type)
                self.edge_types += Counter({edge_type: len(new_neighbor_nodes)})

        if self.node in removed_neighbors:
            for edge_type, removed_neighbor_nodes in removed_neighbors[self.node].items():
                self.remove_edge_model(edge_type)
                self.edge_types -= Counter({edge_type: len(removed_neighbor_nodes)})

    def get_edge_to(self, other_node):
        return DirectedEdge(self.node, other_node)

    def add_edge_model(self, edge_type):
        if self.hyperparams['edge_encoding']:
            if edge_type + '/edge_encoder' not in self.node_modules:
                neighbor_state_length = int(
                    np.sum([len(entity_dims) for entity_dims in
                            self.state[self._get_other_node_type_from_edge(edge_type)].values()]))
                if self.hyperparams['edge_state_combine_method'] == 'pointnet':
                    self.add_submodule(edge_type + '/pointnet_encoder',
                                       model_if_absent=nn.Sequential(
                                           nn.Linear(self.state_length, 2 * self.state_length),
                                           nn.ReLU(),
                                           nn.Linear(2 * self.state_length, 2 * self.state_length),
                                           nn.ReLU()))

                    edge_encoder_input_size = 2 * self.state_length + self.state_length

                elif self.hyperparams['edge_state_combine_method'] == 'attention':
                    self.add_submodule(self.node.type + '/edge_attention_combine',
                                       model_if_absent=TemporallyBatchedAdditiveAttention(
                                           encoder_hidden_state_dim=self.state_length,
                                           decoder_hidden_state_dim=self.state_length))
                    edge_encoder_input_size = self.state_length + neighbor_state_length

                else:
                    edge_encoder_input_size = self.state_length + neighbor_state_length

                self.add_submodule(edge_type + '/edge_encoder',
                                   model_if_absent=nn.LSTM(input_size=edge_encoder_input_size,
                                                           hidden_size=self.hyperparams['enc_rnn_dim_edge'],
                                                           batch_first=True))

    def _get_other_node_type_from_edge(self, edge_type_str):
        n2_type_str = edge_type_str.split('->')[1]
        return NodeType(n2_type_str, self.env.node_type_list.index(n2_type_str) + 1)

    def _get_edge_type_from_str(self, edge_type_str):
        n1_type_str, n2_type_str = edge_type_str.split('->')
        return (NodeType(n1_type_str, self.env.node_type_list.index(n1_type_str) + 1),
                NodeType(n2_type_str, self.env.node_type_list.index(n2_type_str) + 1))

    def remove_edge_model(self, edge_type):
        if self.hyperparams['edge_encoding']:
            if len(self.scene_graph.get_neighbors(self.node, self._get_other_node_type_from_edge(edge_type))) == 0:
                del self.node_modules[edge_type + '/edge_encoder']

    def obtain_encoded_tensors(self,
                               mode,
                               inputs,
                               inputs_st,
                               inputs_np,
                               robot_present_and_future,
                               maps):
        x, x_r_t, y_r = None, None, None
        batch_size = 1

        our_inputs = inputs[self.node]
        our_inputs_st = inputs_st[self.node]

        initial_dynamics = dict()
        initial_dynamics['pos'] = our_inputs[:, 0:2]  # TODO: Generalize
        initial_dynamics['vel'] = our_inputs[:, 2:4]  # TODO: Generalize
        self.dynamic.set_initial_condition(initial_dynamics)

        #########################################
        # Provide basic information to encoders #
        #########################################
        if self.hyperparams['incl_robot_node'] and self.robot is not None:
            robot_present_and_future_st = get_relative_robot_traj(self.env, self.state,
                                                                  our_inputs, robot_present_and_future,
                                                                  self.node.type, self.robot.type)
            x_r_t = robot_present_and_future_st[..., 0, :]
            y_r = robot_present_and_future_st[..., 1:, :]

        ##################
        # Encode History #
        ##################
        node_history_encoded = self.encode_node_history(our_inputs_st)

        ##############################
        # Encode Node Edges per Type #
        ##############################
        total_edge_influence = None
        if self.hyperparams['edge_encoding']:
            node_edges_encoded = list()
            for edge_type in self.edge_types:
                connected_nodes_batched = list()
                edge_masks_batched = list()

                # We get all nodes which are connected to the current node for the current timestep
                connected_nodes_batched.append(self.scene_graph.get_neighbors(self.node,
                                                                              self._get_other_node_type_from_edge(
                                                                                  edge_type)))

                if self.hyperparams['dynamic_edges'] == 'yes':
                    # We get the edge masks for the current node at the current timestep
                    edge_masks_for_node = self.scene_graph.get_edge_scaling(self.node)
                    edge_masks_batched.append(torch.tensor(edge_masks_for_node, dtype=torch.float, device=self.device))

                # Encode edges for given edge type
                encoded_edges_type = self.encode_edge(inputs,
                                                      inputs_st,
                                                      inputs_np,
                                                      edge_type,
                                                      connected_nodes_batched,
                                                      edge_masks_batched)
                node_edges_encoded.append(encoded_edges_type)  # List of [bs/nbs, enc_rnn_dim]

            #####################
            # Encode Node Edges #
            #####################
            total_edge_influence = self.encode_total_edge_influence(mode,
                                                                    node_edges_encoded,
                                                                    node_history_encoded,
                                                                    batch_size)

        self.TD = {'node_history_encoded': node_history_encoded,
                   'total_edge_influence': total_edge_influence}

        ################
        # Map Encoding #
        ################
        if self.hyperparams['use_map_encoding'] and self.node_type in self.hyperparams['map_encoder']:
            if self.node not in maps:
                # This means the node was removed (it is only being kept around because of the edge removal filter).
                me_params = self.hyperparams['map_encoder'][self.node_type]
                self.TD['encoded_map'] = torch.zeros((1, me_params['output_size']))
            else:
                encoded_map = self.node_modules[self.node_type + '/map_encoder'](maps[self.node] * 2. - 1.,
                                                                                 (mode == ModeKeys.TRAIN))
                do = self.hyperparams['map_encoder'][self.node_type]['dropout']
                encoded_map = F.dropout(encoded_map, do, training=(mode == ModeKeys.TRAIN))
                self.TD['encoded_map'] = encoded_map

        ######################################
        # Concatenate Encoder Outputs into x #
        ######################################
        return self.create_encoder_rep(mode, self.TD, x_r_t, y_r)

    def create_encoder_rep(self, mode,
                           TD,
                           robot_present_st,
                           robot_future_st):
        # Unpacking TD
        node_history_encoded = TD['node_history_encoded']
        if self.hyperparams['edge_encoding']:
            total_edge_influence = TD['total_edge_influence']
        if self.hyperparams['use_map_encoding'] and self.node_type in self.hyperparams['map_encoder']:
            encoded_map = TD['encoded_map']

        if (self.hyperparams['incl_robot_node']
                and self.robot is not None
                and robot_future_st is not None
                and robot_present_st is not None):
            robot_future_encoder = self.encode_robot_future(mode, robot_present_st, robot_future_st)

            # Tiling for multiple samples
            # This tiling is done because:
            #   a) we must consider the prediction case where there are many candidate robot future actions,
            #   b) the edge and history encoders are all the same regardless of which candidate future robot action
            #      we're evaluating.
            node_history_encoded = TD['node_history_encoded'].repeat(robot_future_st.size()[0], 1)
            if self.hyperparams['edge_encoding']:
                total_edge_influence = TD['total_edge_influence'].repeat(robot_future_st.size()[0], 1)
            if self.hyperparams['use_map_encoding'] and self.node_type in self.hyperparams['map_encoder']:
                encoded_map = TD['encoded_map'].repeat(robot_future_st.size()[0], 1)

        elif self.hyperparams['incl_robot_node'] and self.robot is not None:
            # Four times because we're trying to mimic a bi-directional RNN's output (which is c and h from both ends).
            robot_future_encoder = torch.zeros([1, 4 * self.hyperparams['enc_rnn_dim_future']], device=self.device)

        x_concat_list = list()

        # Every node has an edge-influence encoder (which could just be zero).
        if self.hyperparams['edge_encoding']:
            x_concat_list.append(total_edge_influence)  # [bs/nbs, 4*enc_rnn_dim]

        # Every node has a history encoder.
        x_concat_list.append(node_history_encoded)  # [bs/nbs, enc_rnn_dim_history]

        if self.hyperparams['incl_robot_node'] and self.robot is not None:
            x_concat_list.append(robot_future_encoder)  # [bs/nbs, 4*enc_rnn_dim_history]

        if self.hyperparams['use_map_encoding'] and self.node_type in self.hyperparams['map_encoder']:
            x_concat_list.append(encoded_map)  # [bs/nbs, CNN output size]

        return torch.cat(x_concat_list, dim=1)

    def encode_node_history(self, inputs_st):
        new_state = torch.unsqueeze(inputs_st, dim=1)  # [bs, 1, state_dim]
        if self.node.type + '/node_history_encoder' not in self.curr_hidden_states:
            outputs, self.curr_hidden_states[self.node.type + '/node_history_encoder'] = self.node_modules[
                self.node.type + '/node_history_encoder'](new_state)
        else:
            outputs, self.curr_hidden_states[self.node.type + '/node_history_encoder'] = self.node_modules[
                self.node.type + '/node_history_encoder'](new_state, self.curr_hidden_states[
                self.node.type + '/node_history_encoder'])

        return outputs[:, 0, :]

    def encode_edge(self, inputs, inputs_st, inputs_np, edge_type, connected_nodes, edge_masks):
        edge_type_tuple = self._get_edge_type_from_str(edge_type)
        edge_states_list = list()  # list of [#of neighbors, max_ht, state_dim]
        neighbor_states = list()

        orig_rel_state = inputs[self.node].cpu().numpy()
        for node in connected_nodes[0]:
            neighbor_state_np = inputs_np[node]

            # Make State relative to node
            _, std = self.env.get_standardize_params(self.state[node.type], node_type=node.type)
            std[0:2] = self.env.attention_radius[edge_type_tuple]

            # TODO: This all makes the unsafe assumption that the first n dims
            #  refer to the same quantities even for different agent types!
            equal_dims = np.min((neighbor_state_np.shape[-1], orig_rel_state.shape[-1]))
            rel_state = np.zeros_like(neighbor_state_np)
            rel_state[..., :equal_dims] = orig_rel_state[..., :equal_dims]
            neighbor_state_np_st = self.env.standardize(neighbor_state_np,
                                                        self.state[node.type],
                                                        node_type=node.type,
                                                        mean=rel_state,
                                                        std=std)

            neighbor_state = torch.tensor(neighbor_state_np_st).float().to(self.device)
            neighbor_states.append(neighbor_state)

        if len(neighbor_states) == 0:  # There are no neighbors for edge type # TODO necessary?
            neighbor_state_length = int(np.sum([len(entity_dims) for entity_dims in self.state[edge_type[1]].values()]))
            edge_states_list.append(torch.zeros((1, 1, neighbor_state_length), device=self.device))
        else:
            edge_states_list.append(torch.stack(neighbor_states, dim=0))

        if self.hyperparams['edge_state_combine_method'] == 'sum':
            # Used in Structural-RNN to combine edges as well.
            op_applied_edge_states_list = list()
            for neighbors_state in edge_states_list:
                op_applied_edge_states_list.append(torch.sum(neighbors_state, dim=0))
            combined_neighbors = torch.stack(op_applied_edge_states_list, dim=0)
            if self.hyperparams['dynamic_edges'] == 'yes':
                # Should now be (bs, time, 1)
                op_applied_edge_mask_list = list()
                for edge_mask in edge_masks:
                    op_applied_edge_mask_list.append(torch.clamp(torch.sum(edge_mask, dim=0, keepdim=True), max=1.))
                combined_edge_masks = torch.stack(op_applied_edge_mask_list, dim=0)

        elif self.hyperparams['edge_state_combine_method'] == 'max':
            # Used in NLP, e.g. max over word embeddings in a sentence.
            op_applied_edge_states_list = list()
            for neighbors_state in edge_states_list:
                op_applied_edge_states_list.append(torch.max(neighbors_state, dim=0))
            combined_neighbors = torch.stack(op_applied_edge_states_list, dim=0)
            if self.hyperparams['dynamic_edges'] == 'yes':
                # Should now be (bs, time, 1)
                op_applied_edge_mask_list = list()
                for edge_mask in edge_masks:
                    op_applied_edge_mask_list.append(torch.clamp(torch.max(edge_mask, dim=0, keepdim=True), max=1.))
                combined_edge_masks = torch.stack(op_applied_edge_mask_list, dim=0)

        elif self.hyperparams['edge_state_combine_method'] == 'mean':
            # Used in NLP, e.g. mean over word embeddings in a sentence.
            op_applied_edge_states_list = list()
            for neighbors_state in edge_states_list:
                op_applied_edge_states_list.append(torch.mean(neighbors_state, dim=0))
            combined_neighbors = torch.stack(op_applied_edge_states_list, dim=0)
            if self.hyperparams['dynamic_edges'] == 'yes':
                # Should now be (bs, time, 1)
                op_applied_edge_mask_list = list()
                for edge_mask in edge_masks:
                    op_applied_edge_mask_list.append(torch.clamp(torch.mean(edge_mask, dim=0, keepdim=True), max=1.))
                combined_edge_masks = torch.stack(op_applied_edge_mask_list, dim=0)

        joint_history = torch.cat([combined_neighbors, torch.unsqueeze(inputs_st[self.node], dim=0)], dim=-1)

        if edge_type + '/edge_encoder' not in self.curr_hidden_states:
            outputs, self.curr_hidden_states[edge_type + '/edge_encoder'] = self.node_modules[
                edge_type + '/edge_encoder'](joint_history)
        else:
            outputs, self.curr_hidden_states[edge_type + '/edge_encoder'] = self.node_modules[
                edge_type + '/edge_encoder'](joint_history, self.curr_hidden_states[edge_type + '/edge_encoder'])

        if self.hyperparams['dynamic_edges'] == 'yes':
            return outputs[:, 0, :] * combined_edge_masks
        else:
            return outputs[:, 0, :]  # [bs, enc_rnn_dim]

    def encoder_forward(self, inputs, inputs_st, inputs_np, robot_present_and_future=None, maps=None):
        # Always predicting with the online model.
        mode = ModeKeys.PREDICT

        self.x = self.obtain_encoded_tensors(mode,
                                             inputs,
                                             inputs_st,
                                             inputs_np,
                                             robot_present_and_future,
                                             maps)
        self.n_s_t0 = inputs_st[self.node]

        self.latent.p_dist = self.p_z_x(mode, self.x)

    # robot_future_st is optional here since you can use the same one from encoder_forward,
    # but if it's given then we'll re-run that part of the model (if the node is adjacent to the robot).
    def decoder_forward(self, prediction_horizon,
                        num_samples,
                        robot_present_and_future=None,
                        z_mode=False,
                        gmm_mode=False,
                        full_dist=False,
                        all_z_sep=False):
        # Always predicting with the online model.
        mode = ModeKeys.PREDICT

        x_nr_t, y_r = None, None
        if (self.hyperparams['incl_robot_node']
                and self.robot is not None
                and robot_present_and_future is not None):
            our_inputs = torch.tensor(self.node.get(np.array([self.node.last_timestep]),
                                                    self.state[self.node.type],
                                                    padding=0.0),
                                      dtype=torch.float,
                                      device=self.device)
            robot_present_and_future_st = get_relative_robot_traj(self.env, self.state,
                                                                  our_inputs, robot_present_and_future,
                                                                  self.node.type, self.robot.type)
            x_nr_t = robot_present_and_future_st[..., 0, :]
            y_r = robot_present_and_future_st[..., 1:, :]
            self.x = self.create_encoder_rep(mode, self.TD, x_nr_t, y_r)
            self.latent.p_dist = self.p_z_x(mode, self.x)

            # Making sure n_s_t0 has the same batch size as x_nr_t
            self.n_s_t0 = self.n_s_t0[[0]].repeat(x_nr_t.size()[0], 1)

        z, num_samples, num_components = self.latent.sample_p(num_samples,
                                                              mode,
                                                              most_likely_z=z_mode,
                                                              full_dist=full_dist,
                                                              all_z_sep=all_z_sep)

        y_dist, our_sampled_future = self.p_y_xz(mode, self.x, x_nr_t, y_r, self.n_s_t0, z,
                                                 prediction_horizon,
                                                 num_samples,
                                                 num_components,
                                                 gmm_mode)

        return y_dist, our_sampled_future
