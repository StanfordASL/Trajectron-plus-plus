import numpy as np
import torch
from model.node_model import MultimodalGenerativeCVAE


class SpatioTemporalGraphCVAEModel(object):
    def __init__(self, model_registrar,
                 hyperparams, log_writer,
                 device):
        super(SpatioTemporalGraphCVAEModel, self).__init__()
        self.hyperparams = hyperparams
        self.log_writer = log_writer
        self.device = device
        self.curr_iter = 0

        self.model_registrar = model_registrar
        self.node_models_dict = dict()
        self.nodes = set()

        self.env = None

        self.min_hl = self.hyperparams['minimum_history_length']
        self.max_hl = self.hyperparams['maximum_history_length']
        self.ph = self.hyperparams['prediction_horizon']
        self.state = self.hyperparams['state']
        self.state_length = dict()
        for type in self.state.keys():
            self.state_length[type] = int(np.sum([len(entity_dims) for entity_dims in self.state[type].values()]))
        self.pred_state = self.hyperparams['pred_state']

    def set_scene_graph(self, env):
        self.env = env

        self.node_models_dict.clear()

        edge_types = env.get_edge_types()

        for node_type in env.NodeType:
            self.node_models_dict[node_type] = MultimodalGenerativeCVAE(env,
                                                                        node_type,
                                                                        self.model_registrar,
                                                                        self.hyperparams,
                                                                        self.device,
                                                                        edge_types,
                                                                        log_writer=self.log_writer)

    def set_curr_iter(self, curr_iter):
        self.curr_iter = curr_iter
        for node_str, model in self.node_models_dict.items():
            model.set_curr_iter(curr_iter)

    def set_annealing_params(self):
        for node_str, model in self.node_models_dict.items():
            model.set_annealing_params()

    def step_annealers(self):
        for node in self.node_models_dict:
            self.node_models_dict[node].step_annealers()

    def get_input(self, scene, timesteps, node_type, min_future_timesteps, max_nodes=None, curve=False):  # Curve is there to resample during training
        inputs = list()
        labels = list()
        first_history_indices = list()
        nodes = list()
        node_scene_graph_batched = list()
        timesteps_in_scene = list()
        nodes_per_ts = scene.present_nodes(timesteps,
                                           type=node_type,
                                           min_history_timesteps=self.min_hl,
                                           min_future_timesteps=min_future_timesteps,
                                           include_robot=not self.hyperparams['incl_robot_node'],
                                           max_nodes=max_nodes,
                                           curve=curve)

        # Get Inputs for each node present in Scene
        for timestep in timesteps:
            if timestep in nodes_per_ts.keys():
                present_nodes = nodes_per_ts[timestep]
                timestep_range = np.array([timestep - self.max_hl, timestep + min_future_timesteps])
                scene_graph_t = scene.get_scene_graph(timestep,
                                                      self.env.attention_radius,
                                                      self.hyperparams['edge_addition_filter'],
                                                      self.hyperparams['edge_removal_filter'])

                for node in present_nodes:
                    timesteps_in_scene.append(timestep)
                    input = node.get(timestep_range, self.state[node.type.name])
                    label = node.get(timestep_range, self.pred_state[node.type.name])
                    first_history_index = (self.max_hl - node.history_points_at(timestep)).clip(0)
                    inputs.append(input)
                    labels.append(label)
                    first_history_indices.append(first_history_index)
                    nodes.append(node)

                    node_scene_graph_batched.append((node, scene_graph_t))

        return inputs, labels, first_history_indices, timesteps_in_scene, node_scene_graph_batched, nodes

    def train_loss(self, scene, timesteps, max_nodes=None):
        losses = dict()
        for node_type in self.env.NodeType:
            losses[node_type] = []

        for node_type in self.env.NodeType:
            # Get Input data for node type and given timesteps
            (inputs,
             labels,
             first_history_indices,
             timesteps_in_scene,
             node_scene_graph_batched, _) = self.get_input(scene,
                                                           timesteps,
                                                           node_type,
                                                           self.ph,
                                                           max_nodes=max_nodes,
                                                           curve=True)  # Curve is there to resample during training

            # There are no nodes of type present for timestep
            if len(inputs) == 0:
                continue

            uniform_t = self.max_hl

            inputs = np.array(inputs)
            labels = np.array(labels)

            # Vehicles are rotated such that the x axis is lateral
            if node_type == self.env.NodeType.VEHICLE:
                # transform x y to ego.
                pos = inputs[..., 0:2]
                pos_org = pos.copy()
                vel = inputs[..., 2:4]
                acc = inputs[..., 5:7]
                heading = inputs[:, uniform_t, -1]
                rot_mat = np.zeros((pos.shape[0], pos.shape[1], 3, 3))
                rot_mat[:, :, 0, 0] = np.cos(heading)[:, np.newaxis]
                rot_mat[:, :, 0, 1] = np.sin(heading)[:, np.newaxis]
                rot_mat[:, :, 1, 0] = -np.sin(heading)[:, np.newaxis]
                rot_mat[:, :, 1, 1] = np.cos(heading)[:, np.newaxis]
                rot_mat[:, :, 2, 2] = 1.

                pos = pos - pos[:, uniform_t, np.newaxis, :]

                pos_with_one = np.ones((pos.shape[0], pos.shape[1], 3, 1))
                pos_with_one[:, :, :2] = pos[..., np.newaxis]
                pos_rot = np.squeeze(rot_mat @ pos_with_one, axis=-1)[..., :2]

                vel_with_one = np.ones((vel.shape[0], vel.shape[1], 3, 1))
                vel_with_one[:, :, :2] = vel[..., np.newaxis]
                vel_rot = np.squeeze(rot_mat @ vel_with_one, axis=-1)[..., :2]

                acc_with_one = np.ones((acc.shape[0], acc.shape[1], 3, 1))
                acc_with_one[:, :, :2] = acc[..., np.newaxis]
                acc_rot = np.squeeze(rot_mat @ acc_with_one, axis=-1)[..., :2]

                inputs[..., 0:2] = pos_rot
                inputs[..., 2:4] = vel_rot
                inputs[..., 5:7] = acc_rot

                l_vel_with_one = np.ones((labels.shape[0], labels.shape[1], 3, 1))
                l_vel_with_one[:, :, :2] = labels[..., np.newaxis]
                labels = np.squeeze(rot_mat @ l_vel_with_one, axis=-1)[..., :2]

            # Standardize, Position is standardized relative to current pos and attention_radius for node_type-node_type
            _, std = self.env.get_standardize_params(self.state[node_type.name], node_type=node_type)
            # std[0:2] = self.env.attention_radius[(node_type, node_type)]
            rel_state = np.array(inputs)[:, uniform_t]
            rel_state = np.hstack((rel_state, np.zeros_like(rel_state)))
            rel_state = np.expand_dims(rel_state, 1)
            std = np.tile(std, 2)
            inputs = np.tile(inputs, 2)
            inputs[..., self.state_length[node_type.name]:self.state_length[node_type.name]+2] = 0.
            inputs_st = self.env.standardize(inputs,
                                             self.state[node_type.name],
                                             mean=rel_state,
                                             std=std,
                                             node_type=node_type)
            labels_st = self.env.standardize(labels, self.pred_state[node_type.name], node_type=node_type)

            if node_type == self.env.NodeType.VEHICLE:
                inputs[..., 0:2] = pos_org

            # Convert to torch tensors
            inputs = torch.tensor(inputs).float().to(self.device)
            inputs_st = torch.tensor(inputs_st).float().to(self.device)
            first_history_indices = torch.tensor(first_history_indices).float().to(self.device).long()
            labels = torch.tensor(labels).float().to(self.device)
            labels_st = torch.tensor(labels_st).float().to(self.device)

            # Run forward pass
            model = self.node_models_dict[node_type]
            loss = model.train_loss(inputs,
                                    inputs_st,
                                    first_history_indices,
                                    labels,
                                    labels_st,
                                    scene,
                                    node_scene_graph_batched,
                                    timestep=uniform_t,
                                    timesteps_in_scene=timesteps_in_scene,
                                    prediction_horizon=self.ph)

            losses[node_type].append(loss)

        for node_type in self.env.NodeType:
            losses[node_type] = torch.mean(torch.stack(losses[node_type])) if len(losses[node_type]) > 0 else None
        return losses

    def eval_loss(self, scene, timesteps, max_nodes=None):
        losses = dict()
        for node_type in self.env.NodeType:
            losses[node_type] = {'nll_q_is': list(), 'nll_p': list(), 'nll_exact': list(), 'nll_sampled': list()}
        for node_type in self.env.NodeType:
            # Get Input data for node type and given timesteps
            (inputs,
             labels,
             first_history_indices,
             timesteps_in_scene,
             node_scene_graph_batched, _) = self.get_input(scene, timesteps, node_type, self.ph, max_nodes=max_nodes)

            # There are no nodes of type present for timestep
            if len(inputs) == 0:
                continue

            uniform_t = self.max_hl

            inputs = np.array(inputs)
            labels = np.array(labels)

            # Vehicles are rotated such that the x axis is lateral
            if node_type == self.env.NodeType.VEHICLE:
                # transform x y to ego.
                pos = inputs[..., 0:2]
                pos_org = pos.copy()
                vel = inputs[..., 2:4]
                acc = inputs[..., 5:7]
                heading = inputs[:, uniform_t, -1]
                rot_mat = np.zeros((pos.shape[0], pos.shape[1], 3, 3))
                rot_mat[:, :, 0, 0] = np.cos(heading)[:, np.newaxis]
                rot_mat[:, :, 0, 1] = np.sin(heading)[:, np.newaxis]
                rot_mat[:, :, 1, 0] = -np.sin(heading)[:, np.newaxis]
                rot_mat[:, :, 1, 1] = np.cos(heading)[:, np.newaxis]
                rot_mat[:, :, 2, 2] = 1.

                pos = pos - pos[:, uniform_t, np.newaxis, :]

                pos_with_one = np.ones((pos.shape[0], pos.shape[1], 3, 1))
                pos_with_one[:, :, :2] = pos[..., np.newaxis]
                pos_rot = np.squeeze(rot_mat @ pos_with_one, axis=-1)[..., :2]

                vel_with_one = np.ones((vel.shape[0], vel.shape[1], 3, 1))
                vel_with_one[:, :, :2] = vel[..., np.newaxis]
                vel_rot = np.squeeze(rot_mat @ vel_with_one, axis=-1)[..., :2]

                acc_with_one = np.ones((acc.shape[0], acc.shape[1], 3, 1))
                acc_with_one[:, :, :2] = acc[..., np.newaxis]
                acc_rot = np.squeeze(rot_mat @ acc_with_one, axis=-1)[..., :2]

                inputs[..., 0:2] = pos_rot
                inputs[..., 2:4] = vel_rot
                inputs[..., 5:7] = acc_rot

                l_vel_with_one = np.ones((labels.shape[0], labels.shape[1], 3, 1))
                l_vel_with_one[:, :, :2] = labels[..., np.newaxis]
                labels = np.squeeze(rot_mat @ l_vel_with_one, axis=-1)[..., :2]

            # Standardize, Position is standardized relative to current pos and attention_radius for node_type-node_type
            _, std = self.env.get_standardize_params(self.state[node_type.name], node_type=node_type)
            rel_state = np.array(inputs)[:, uniform_t]
            rel_state = np.hstack((rel_state, np.zeros_like(rel_state)))
            rel_state = np.expand_dims(rel_state, 1)
            std = np.tile(std, 2)
            inputs = np.tile(inputs, 2)
            inputs[..., self.state_length[node_type.name]:self.state_length[node_type.name]+2] = 0.
            inputs_st = self.env.standardize(inputs,
                                             self.state[node_type.name],
                                             mean=rel_state,
                                             std=std,
                                             node_type=node_type)
            labels_st = self.env.standardize(labels, self.pred_state[node_type.name], node_type=node_type)

            if node_type == self.env.NodeType.VEHICLE:
                inputs[..., 0:2] = pos_org

            # Convert to torch tensors
            inputs = torch.tensor(inputs).float().to(self.device)
            inputs_st = torch.tensor(inputs_st).float().to(self.device)
            first_history_indices = torch.tensor(first_history_indices).float().to(self.device).long()
            labels = torch.tensor(labels).float().to(self.device)
            labels_st = torch.tensor(labels_st).float().to(self.device)

            # Run forward pass
            model = self.node_models_dict[node_type]
            (nll_q_is, nll_p, nll_exact, nll_sampled) = model.eval_loss(inputs,
                                                                        inputs_st,
                                                                        first_history_indices,
                                                                        labels,
                                                                        labels_st,
                                                                        scene,
                                                                        node_scene_graph_batched,
                                                                        timestep=uniform_t,
                                                                        timesteps_in_scene=timesteps_in_scene,
                                                                        prediction_horizon=self.ph)

            if nll_q_is is not None:
                losses[node_type]['nll_q_is'].append(nll_q_is.cpu().numpy())
                losses[node_type]['nll_p'].append(nll_p.cpu().numpy())
                losses[node_type]['nll_exact'].append(nll_exact.cpu().numpy())
                losses[node_type]['nll_sampled'].append(nll_sampled.cpu().numpy())

        return losses

    def predict(self,
                scene,
                timesteps,
                ph,
                num_samples_z=1,
                num_samples_gmm=1,
                min_future_timesteps=0,
                most_likely_z=False,
                most_likely_gmm=False,
                all_z=False,
                max_nodes=None):

        predictions_dict = {}
        for node_type in self.env.NodeType:
            # Get Input data for node type and given timesteps
            (inputs,
             labels,
             first_history_indices,
             timesteps_in_scene,
             node_scene_graph_batched,
             nodes) = self.get_input(scene, timesteps, node_type, min_future_timesteps, max_nodes=max_nodes)

            # There are no nodes of type present for timestep
            if len(inputs) == 0:
                continue

            uniform_t = self.max_hl

            inputs = np.array(inputs)
            labels = np.array(labels)

            # Vehicles are rotated such that the x axis is lateral
            if node_type == self.env.NodeType.VEHICLE:
                # transform x y to ego.
                pos = inputs[..., 0:2]
                pos_org = pos.copy()
                vel = inputs[..., 2:4]
                acc = inputs[..., 5:7]
                heading = inputs[:, uniform_t, -1]
                rot_mat = np.zeros((pos.shape[0], pos.shape[1], 3, 3))
                rot_mat[:, :, 0, 0] = np.cos(heading)[:, np.newaxis]
                rot_mat[:, :, 0, 1] = np.sin(heading)[:, np.newaxis]
                rot_mat[:, :, 1, 0] = -np.sin(heading)[:, np.newaxis]
                rot_mat[:, :, 1, 1] = np.cos(heading)[:, np.newaxis]
                rot_mat[:, :, 2, 2] = 1.

                pos = pos - pos[:, uniform_t, np.newaxis, :]

                pos_with_one = np.ones((pos.shape[0], pos.shape[1], 3, 1))
                pos_with_one[:, :, :2] = pos[..., np.newaxis]
                pos_rot = np.squeeze(rot_mat @ pos_with_one, axis=-1)[..., :2]

                vel_with_one = np.ones((vel.shape[0], vel.shape[1], 3, 1))
                vel_with_one[:, :, :2] = vel[..., np.newaxis]
                vel_rot = np.squeeze(rot_mat @ vel_with_one, axis=-1)[..., :2]

                acc_with_one = np.ones((acc.shape[0], acc.shape[1], 3, 1))
                acc_with_one[:, :, :2] = acc[..., np.newaxis]
                acc_rot = np.squeeze(rot_mat @ acc_with_one, axis=-1)[..., :2]

                inputs[..., 0:2] = pos_rot
                inputs[..., 2:4] = vel_rot
                inputs[..., 5:7] = acc_rot

                l_vel_with_one = np.ones((labels.shape[0], labels.shape[1], 3, 1))
                l_vel_with_one[:, :, :2] = labels[..., np.newaxis]
                labels = np.squeeze(rot_mat @ l_vel_with_one, axis=-1)[..., :2]

            # Standardize, Position is standardized relative to current pos and attention_radius for node_type-node_type
            _, std = self.env.get_standardize_params(self.state[node_type.name], node_type=node_type)
            rel_state = np.array(inputs)[:, uniform_t]
            rel_state = np.hstack((rel_state, np.zeros_like(rel_state)))
            rel_state = np.expand_dims(rel_state, 1)
            std = np.tile(std, 2)
            inputs = np.tile(inputs, 2)
            inputs[..., self.state_length[node_type.name]:self.state_length[node_type.name]+2] = 0.
            inputs_st = self.env.standardize(inputs,
                                             self.state[node_type.name],
                                             mean=rel_state,
                                             std=std,
                                             node_type=node_type)
            labels_st = self.env.standardize(labels, self.pred_state[node_type.name], node_type=node_type)

            if node_type == self.env.NodeType.VEHICLE:
                inputs[..., 0:2] = pos_org

            # Convert to torch tensors
            inputs = torch.tensor(inputs).float().to(self.device)
            inputs_st = torch.tensor(inputs_st).float().to(self.device)
            first_history_indices = torch.tensor(first_history_indices).float().to(self.device).long()
            labels = torch.tensor(labels).float().to(self.device)
            labels_st = torch.tensor(labels_st).float().to(self.device)

            # Run forward pass
            model = self.node_models_dict[node_type]
            predictions = model.predict(inputs,
                                        inputs_st,
                                        labels,
                                        labels_st,
                                        first_history_indices,
                                        scene,
                                        node_scene_graph_batched,
                                        timestep=uniform_t,
                                        timesteps_in_scene=timesteps_in_scene,
                                        prediction_horizon=ph,
                                        num_samples_z=num_samples_z,
                                        num_samples_gmm=num_samples_gmm,
                                        most_likely_z=most_likely_z,
                                        most_likely_gmm=most_likely_gmm,
                                        all_z=all_z)

            predictions_uns = self.env.unstandardize(predictions.cpu().detach().numpy(),
                                                     self.pred_state[node_type.name],
                                                     node_type)

            # Vehicles are rotated such that the x axis is lateral. For output rotation has to be reversed
            if node_type == self.env.NodeType.VEHICLE:
                heading = inputs.cpu().detach().numpy()[:, uniform_t, -1]
                rot_mat = np.zeros((predictions_uns.shape[0],
                                    predictions_uns.shape[1],
                                    predictions_uns.shape[2],
                                    predictions_uns.shape[3], 3, 3))
                rot_mat[:, :, :, :, 0, 0] = np.cos(-heading)[:, np.newaxis]
                rot_mat[:, :, :, :, 0, 1] = np.sin(-heading)[:, np.newaxis]
                rot_mat[:, :, :, :, 1, 0] = -np.sin(-heading)[:, np.newaxis]
                rot_mat[:, :, :, :, 1, 1] = np.cos(-heading)[:, np.newaxis]
                rot_mat[:, :, :, :, 2, 2] = 1.

                p_vel_with_one = np.ones((predictions_uns.shape[0],
                                          predictions_uns.shape[1],
                                          predictions_uns.shape[2],
                                          predictions_uns.shape[3], 3, 1))
                p_vel_with_one[:, :, :, :, :2] = predictions_uns[..., np.newaxis]
                predictions_uns = np.squeeze(rot_mat @ p_vel_with_one, axis=-1)[..., :2]

            # Assign predictions to node
            for i, ts in enumerate(timesteps_in_scene):
                if not ts in predictions_dict.keys():
                    predictions_dict[ts] = dict()
                predictions_dict[ts][nodes[i]] = predictions_uns[:, :, i]

        return predictions_dict

