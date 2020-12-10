import torch
import numpy as np
from collections import Counter
from model.trajectron import Trajectron
from model.online.online_mgcvae import OnlineMultimodalGenerativeCVAE
from model.model_utils import ModeKeys
from environment import RingBuffer, TemporalSceneGraph, SceneGraph, derivative_of


class OnlineTrajectron(Trajectron):
    def __init__(self, model_registrar,
                 hyperparams, device):
        super(OnlineTrajectron, self).__init__(model_registrar=model_registrar,
                                               hyperparams=hyperparams,
                                               log_writer=False,
                                               device=device)
        self.node_data = dict()
        self.scene_graph = None
        self.RING_CAPACITY = max(len(self.hyperparams['edge_removal_filter']),
                                 len(self.hyperparams['edge_addition_filter']),
                                 self.hyperparams['maximum_history_length']) + 1
        self.rel_states = dict()
        self.removed_nodes = Counter()

    def __repr__(self):
        return f"OnlineTrajectron(# nodes: {len(self.nodes)}, device: {self.device}, hyperparameters: {str(self.hyperparams)}) "

    def _add_node_model(self, node):
        if node in self.nodes:
            raise ValueError('%s was already added to this graph!' % str(node))

        self.nodes.add(node)
        self.node_models_dict[node] = OnlineMultimodalGenerativeCVAE(self.env,
                                                                     node,
                                                                     self.model_registrar,
                                                                     self.hyperparams,
                                                                     self.device)

    def update_removed_nodes(self):
        for node in list(self.removed_nodes.keys()):
            if self.removed_nodes[node] >= len(self.hyperparams['edge_removal_filter']):
                del self.node_data[node]
                del self.removed_nodes[node]

    def _remove_node_model(self, node):
        if node not in self.nodes:
            raise ValueError('%s is not in this graph!' % str(node))

        self.nodes.remove(node)
        del self.node_models_dict[node]

    def set_environment(self, env, init_timestep=0):
        self.env = env
        self.scene_graph = SceneGraph(edge_radius=self.env.attention_radius)
        self.nodes.clear()
        self.node_data.clear()
        self.node_models_dict.clear()

        # Fast-forwarding ourselves to the initial timestep, without running any of the underlying models.
        for timestep in range(init_timestep + 1):
            self.incremental_forward(self.env.scenes[0].get_clipped_input_dict(timestep, self.hyperparams['state']),
                                     maps=None, run_models=False)

    def incremental_forward(self, new_inputs_dict,
                            maps,
                            prediction_horizon=0,
                            num_samples=0,
                            robot_present_and_future=None,
                            z_mode=False,
                            gmm_mode=False,
                            full_dist=False,
                            all_z_sep=False,
                            run_models=True):
        # The way this function works is by appending the new datapoints to the
        # ends of each of the LSTMs in the graph. Then, we recalculate the
        # encoder's output vector h_x and feed that into the decoder to sample new outputs.
        mode = ModeKeys.PREDICT

        # No grad since we're predicting always, as evidenced by the line above.
        with torch.no_grad():
            for node, new_input in new_inputs_dict.items():
                if node not in self.node_data:
                    self.node_data[node] = RingBuffer(capacity=self.RING_CAPACITY,
                                                      dtype=(float, sum(len(self.state[node.type][k])
                                                                        for k in self.state[node.type])))
                self.node_data[node].append(new_input)

                if node in self.removed_nodes:
                    del self.removed_nodes[node]

            # Nodes in self.node_data that aren't in new_inputs_dict were just removed.
            newly_removed_nodes = (set(self.node_data.keys()) - set(self.removed_nodes.keys())) - set(
                new_inputs_dict.keys())

            # We update self.removed_nodes with the newly removed nodes as well as all existing removed nodes to get
            # the time since their last removal increased by one.
            self.removed_nodes.update(newly_removed_nodes | set(self.removed_nodes.keys()))

            # For any nodes that are older than the length of the edge_removal_filter, we can safely clear their data.
            self.update_removed_nodes()

            # Any remaining removed nodes that aren't yet old enough for data clearing simply have NaNs appended so
            # that when it's passed through the LSTMs, the hidden state keeps propagating but the input plays no role
            # (the NaNs get converted to zeros later on).
            for node in self.removed_nodes:
                self.node_data[node].append(np.full((1, self.node_data[node].shape[1]), np.nan))

            for node in self.node_data:
                node.overwrite_data(self.node_data[node], None,
                                    forward_in_time_on_next_overwrite=(self.node_data[node].shape[0]
                                                                       == self.RING_CAPACITY))

            temp_scene_dict = {k: v[:, 0:2] for k, v in self.node_data.items()}
            if not temp_scene_dict:
                new_scene_graph = SceneGraph(edge_radius=self.env.attention_radius)
            else:
                new_scene_graph = TemporalSceneGraph.create_from_temp_scene_dict(
                    temp_scene_dict,
                    self.env.attention_radius,
                    duration=self.RING_CAPACITY,
                    edge_addition_filter=self.hyperparams['edge_addition_filter'],
                    edge_removal_filter=self.hyperparams['edge_removal_filter'],
                    online=True).to_scene_graph(t=self.RING_CAPACITY - 1)

            if self.hyperparams['dynamic_edges'] == 'yes':
                new_nodes, removed_nodes, new_neighbors, removed_neighbors = new_scene_graph - self.scene_graph

                # Aside from updating the scene graph, this for loop updates the graph model
                # structure of all affected nodes.
                not_removed_nodes = [node for node in self.nodes if node not in removed_nodes]
                self.scene_graph = new_scene_graph
                for node in not_removed_nodes:
                    self.node_models_dict[node].update_graph(new_scene_graph, new_neighbors, removed_neighbors)

                # These next 2 for loops add or remove entire node models.
                for node in new_nodes:
                    if (node.is_robot and self.hyperparams['incl_robot_node']) or node.type not in self.pred_state.keys():
                        # Only deal with Models for NodeTypes we want to predict
                        continue

                    self._add_node_model(node)
                    self.node_models_dict[node].update_graph(new_scene_graph, new_neighbors, removed_neighbors)

                for node in removed_nodes:
                    if (node.is_robot and self.hyperparams['incl_robot_node']) or node.type not in self.pred_state.keys():
                        continue

                    self._remove_node_model(node)

            # This actually updates the node models with the newly observed data.
            if run_models:
                inputs = dict()
                inputs_st = dict()
                inputs_np = dict()

                iter_list = list(self.node_models_dict.keys()) + [node for node in new_inputs_dict
                                                                    if node.type not in self.pred_state.keys()]
                if self.env.scenes[0].robot is not None:
                    iter_list.append(self.env.scenes[0].robot)

                for node in iter_list:
                    input_np = node.get(np.array([node.last_timestep, node.last_timestep]), self.state[node.type])

                    _, std = self.env.get_standardize_params(self.state[node.type.name], node.type)
                    std[0:2] = self.env.attention_radius[(node.type, node.type)]
                    rel_state = np.zeros_like(input_np)
                    rel_state[:, 0:2] = input_np[:, 0:2]
                    input_st = self.env.standardize(input_np,
                                                    self.state[node.type.name],
                                                    node.type,
                                                    mean=rel_state)
                    self.rel_states[node] = rel_state

                    # Converting NaNs to zeros.
                    input_np[np.isnan(input_np)] = 0
                    input_st[np.isnan(input_st)] = 0

                    # Convert to torch tensors
                    inputs[node] = torch.tensor(input_np, dtype=torch.float, device=self.device)
                    inputs_st[node] = torch.tensor(input_st, dtype=torch.float, device=self.device)
                    inputs_np[node] = input_np

                # We want tensors of shape (1, ph + 1, state_dim) where the first 1 is the batch size.
                if (self.hyperparams['incl_robot_node']
                        and self.env.scenes[0].robot is not None
                        and robot_present_and_future is not None):
                    if len(robot_present_and_future.shape) == 2:
                        robot_present_and_future = robot_present_and_future[np.newaxis, :]

                    assert robot_present_and_future.shape[1] == prediction_horizon + 1
                    robot_present_and_future = torch.tensor(robot_present_and_future,
                                                            dtype=torch.float, device=self.device)

                for node in self.node_models_dict:
                    self.node_models_dict[node].encoder_forward(inputs,
                                                                inputs_st,
                                                                inputs_np,
                                                                robot_present_and_future,
                                                                maps)

                # If num_predicted_timesteps or num_samples == 0 then do not run the decoder at all,
                # just update the encoder LSTMs.
                if prediction_horizon == 0 or num_samples == 0:
                    return

                return self.sample_model(prediction_horizon,
                                         num_samples,
                                         robot_present_and_future=robot_present_and_future,
                                         z_mode=z_mode,
                                         gmm_mode=gmm_mode,
                                         full_dist=full_dist,
                                         all_z_sep=all_z_sep)

    def _run_decoder(self, node,
                     num_predicted_timesteps,
                     num_samples,
                     robot_present_and_future=None,
                     z_mode=False,
                     gmm_mode=False,
                     full_dist=False,
                     all_z_sep=False):
        model = self.node_models_dict[node]
        prediction_dist, predictions_uns = model.decoder_forward(num_predicted_timesteps,
                                                                 num_samples,
                                                                 robot_present_and_future=robot_present_and_future,
                                                                 z_mode=z_mode,
                                                                 gmm_mode=gmm_mode,
                                                                 full_dist=full_dist,
                                                                 all_z_sep=all_z_sep)

        predictions_np = predictions_uns.cpu().detach().numpy()

        # Return will be of shape (batch_size, num_samples, num_predicted_timesteps, 2)
        return prediction_dist, np.transpose(predictions_np, (1, 0, 2, 3))

    def sample_model(self, num_predicted_timesteps,
                     num_samples,
                     robot_present_and_future=None,
                     z_mode=False,
                     gmm_mode=False,
                     full_dist=False,
                     all_z_sep=False):
        # Just start from the encoder output (minus the
        # robot future) and get num_samples of
        # num_predicted_timesteps-length trajectories.
        if num_predicted_timesteps == 0 or num_samples == 0:
            return

        mode = ModeKeys.PREDICT

        # We want tensors of shape (1, ph + 1, state_dim) where the first 1 is the batch size.
        if self.hyperparams['incl_robot_node'] and self.env.scenes[
            0].robot is not None and robot_present_and_future is not None:
            if len(robot_present_and_future.shape) == 2:
                robot_present_and_future = robot_present_and_future[np.newaxis, :]

            assert robot_present_and_future.shape[1] == num_predicted_timesteps + 1

        # No grad since we're predicting always, as evidenced by the line above.
        with torch.no_grad():
            predictions_dict = dict()
            prediction_dists = dict()
            for node in set(self.nodes) - set(self.removed_nodes.keys()):
                if node.is_robot:
                    continue

                prediction_dists[node], predictions_dict[node] = self._run_decoder(node, num_predicted_timesteps,
                                                                                   num_samples,
                                                                                   robot_present_and_future,
                                                                                   z_mode,
                                                                                   gmm_mode,
                                                                                   full_dist,
                                                                                   all_z_sep)

        return prediction_dists, predictions_dict

    def forward(self, init_env,
                init_timestep,
                input_dicts,  # After the initial environment
                num_predicted_timesteps,
                num_samples,
                robot_present_and_future=None,
                z_mode=False,
                gmm_mode=False,
                full_dist=False,
                all_z_sep=False):
        # This is the standard forward prediction function,
        # if you have some historical data and just want to
        # predict forward some number of timesteps.

        # Setting us back to the initial scene graph we had.
        self.set_environment(init_env, init_timestep)

        # Looping through and applying updates to the model.
        for i in range(len(input_dicts)):
            self.incremental_forward(input_dicts[i])

        return self.sample_model(num_predicted_timesteps,
                                 num_samples,
                                 robot_present_and_future=robot_present_and_future,
                                 z_mode=z_mode,
                                 gmm_mode=gmm_mode,
                                 full_dist=full_dist,
                                 all_z_sep=all_z_sep)
