import warnings
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model.components import *
from model.model_utils import *


class MultimodalGenerativeCVAE(object):
    def __init__(self,
                 env,
                 node_type,
                 model_registrar,
                 hyperparams,
                 device,
                 edge_types,
                 log_writer=None):
        self.env = env
        self.node_type = node_type.name
        self.model_registrar = model_registrar
        self.log_writer = log_writer
        self.device = device
        self.edge_types = [edge_type for edge_type in edge_types if edge_type[0] is node_type]
        self.curr_iter = 0

        self.node_modules = dict()
        self.hyperparams = hyperparams

        self.min_hl = self.hyperparams['minimum_history_length']
        self.max_hl = self.hyperparams['maximum_history_length']
        self.ph = self.hyperparams['prediction_horizon']
        self.state = self.hyperparams['state']
        self.pred_state = self.hyperparams['pred_state'][node_type.name]
        self.state_length = int(np.sum([len(entity_dims) for entity_dims in self.state[node_type.name].values()])) * 2 # We have the relative and absolute state
        self.robot_state_length = int(np.sum([len(entity_dims) for entity_dims in self.state['VEHICLE'].values()])) # TODO VEHICLE is hard coded for now
        self.pred_state_length = int(np.sum([len(entity_dims) for entity_dims in self.pred_state.values()]))

        edge_types_str = [env.edge_type_str(edge_type) for edge_type in self.edge_types]

        self.create_graphical_model(edge_types_str)

    def set_curr_iter(self, curr_iter):
        self.curr_iter = curr_iter

    def add_submodule(self, name, model_if_absent):
        self.node_modules[name] = self.model_registrar.get_model(name, model_if_absent)

    def clear_submodules(self):
        self.node_modules.clear()

    def create_graphical_model(self, edge_types):
        self.clear_submodules()

        ############################
        #   Node History Encoder   #
        ############################
        self.add_submodule(self.node_type + '/node_history_encoder',
                           model_if_absent=nn.LSTM(input_size=self.state_length,
                                                   hidden_size=self.hyperparams['enc_rnn_dim_history'],
                                                   batch_first=True))

        ###########################
        #   Node Future Encoder   #
        ###########################
        # We'll create this here, but then later check if in training mode.
        # Based on that, we'll factor this into the computation graph (or not).
        self.add_submodule(self.node_type + '/node_future_encoder',
                           model_if_absent=nn.LSTM(input_size=self.pred_state_length,
                                                   hidden_size=self.hyperparams['enc_rnn_dim_future'],
                                                   bidirectional=True,
                                                   batch_first=True))
        # These are related to how you initialize states for the node future encoder.
        self.add_submodule(self.node_type + '/node_future_encoder/initial_h',
                           model_if_absent=nn.Linear(self.state_length,
                                                     self.hyperparams['enc_rnn_dim_future']))
        self.add_submodule(self.node_type + '/node_future_encoder/initial_c',
                           model_if_absent=nn.Linear(self.state_length,
                                                     self.hyperparams['enc_rnn_dim_future']))

        ############################
        #   Robot Future Encoder   #
        ############################
        # We'll create this here, but then later check if we're next to the robot.
        # Based on that, we'll factor this into the computation graph (or not).
        if self.hyperparams['incl_robot_node']:
            self.add_submodule('robot_future_encoder',
                               model_if_absent=nn.LSTM(input_size=self.robot_state_length ,
                                                       hidden_size=self.hyperparams['enc_rnn_dim_future'],
                                                       bidirectional=True,
                                                       batch_first=True))
            # These are related to how you initialize states for the robot future encoder.
            self.add_submodule('robot_future_encoder/initial_h',
                               model_if_absent=nn.Linear(self.robot_state_length,
                                                         self.hyperparams['enc_rnn_dim_future']))
            self.add_submodule('robot_future_encoder/initial_c',
                               model_if_absent=nn.Linear(self.robot_state_length,
                                                         self.hyperparams['enc_rnn_dim_future']))

        #####################
        #   Edge Encoders   #
        #####################
        # print('create_graphical_model', self.node)
        # print('create_graphical_model', self.neighbors_via_edge_type)
        for edge_type in edge_types:
            neighbor_state_length = int(np.sum([len(entity_dims) for entity_dims in self.state[edge_type.split('-')[1]].values()]))
            if self.hyperparams['edge_state_combine_method'] == 'pointnet':
                self.add_submodule(edge_type + '/pointnet_encoder',
                                   model_if_absent=nn.Sequential(
                                       nn.Linear(self.state_length, 2 * self.state_length),
                                       nn.ReLU(),
                                       nn.Linear(2 * self.state_length, 2 * self.state_length),
                                       nn.ReLU()))

                edge_encoder_input_size = 2 * self.state_length + self.state_length

            elif self.hyperparams['edge_state_combine_method'] == 'attention':
                self.add_submodule(self.node_type + '/edge_attention_combine',
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

        ##############################
        #   Edge Influence Encoder   #
        ##############################
        # NOTE: The edge influence encoding happens during calls
        # to forward or incremental_forward, so we don't create
        # a model for it here for the max and sum variants.
        if self.hyperparams['edge_influence_combine_method'] == 'bi-rnn':
            self.add_submodule(self.node_type + '/edge_influence_encoder',
                               model_if_absent=nn.LSTM(input_size=self.hyperparams['enc_rnn_dim_edge'],
                                                       hidden_size=self.hyperparams['enc_rnn_dim_edge_influence'],
                                                       bidirectional=True,
                                                       batch_first=True))

            # Four times because we're trying to mimic a bi-directional
            # LSTM's output (which, here, is c and h from both ends).
            self.eie_output_dims = 4 * self.hyperparams['enc_rnn_dim_edge_influence']

        elif self.hyperparams['edge_influence_combine_method'] == 'attention':
            # Chose additive attention because of https://arxiv.org/pdf/1703.03906.pdf
            # We calculate an attention context vector using the encoded edges as the "encoder" (that we attend _over_)
            # and the node history encoder representation as the "decoder state" (that we attend _on_).
            self.add_submodule(self.node_type + '/edge_influence_encoder',
                               model_if_absent=AdditiveAttention(
                                   encoder_hidden_state_dim=self.hyperparams['enc_rnn_dim_edge_influence'],
                                   decoder_hidden_state_dim=self.hyperparams['enc_rnn_dim_history']))

            self.eie_output_dims = self.hyperparams['enc_rnn_dim_edge_influence']

        ###################
        #   Map Encoder   #
        ###################
        if self.hyperparams['use_map_encoding']:
            self.add_submodule(self.node_type + '/map_encoder',
                               model_if_absent=CNNMapEncoder(input_size=self.hyperparams['map_context'],
                                                             hidden_size=self.hyperparams['map_enc_hidden_size'],
                                                             output_size=self.hyperparams['map_enc_output_size']))

        ################################
        #   Discrete Latent Variable   #
        ################################
        self.latent = DiscreteLatent(self.hyperparams, self.device)

        ######################################################################
        #   Various Fully-Connected Layers from Encoder to Latent Variable   #
        ######################################################################
        #       Edge Influence Encoder         Node History Encoder
        x_size = self.eie_output_dims + self.hyperparams['enc_rnn_dim_history']
        if self.hyperparams['incl_robot_node']:
            #              Future Conditional Encoder
            x_size += 4 * self.hyperparams['enc_rnn_dim_future']

        if self.hyperparams['use_map_encoding']:
            #                    Map Encoder
            x_size += self.hyperparams['map_enc_output_size']

        z_size = self.hyperparams['N'] * self.hyperparams['K']

        if self.hyperparams['p_z_x_MLP_dims'] is not None:
            self.add_submodule(self.node_type + '/p_z_x',
                               model_if_absent=nn.Linear(x_size, self.hyperparams['p_z_x_MLP_dims']))
            hx_size = self.hyperparams['p_z_x_MLP_dims']
        else:
            hx_size = x_size

        self.add_submodule(self.node_type + '/hx_to_z',
                           model_if_absent=nn.Linear(hx_size, self.latent.z_dim))

        if self.hyperparams['q_z_xy_MLP_dims'] is not None:
            self.add_submodule(self.node_type + '/q_z_xy',
                               #                                           Node Future Encoder
                               model_if_absent=nn.Linear(x_size + 4 * self.hyperparams['enc_rnn_dim_future'],
                                                         self.hyperparams['q_z_xy_MLP_dims']))
            hxy_size = self.hyperparams['q_z_xy_MLP_dims']
        else:
            #                           Node Future Encoder
            hxy_size = x_size + 4 * self.hyperparams['enc_rnn_dim_future']

        self.add_submodule(self.node_type + '/hxy_to_z',
                           model_if_absent=nn.Linear(hxy_size, self.latent.z_dim))

        ####################
        #   Decoder LSTM   #
        ####################
        if self.hyperparams['incl_robot_node']:
            decoder_input_dims = self.pred_state_length + self.robot_state_length + z_size + x_size
        else:
            decoder_input_dims = self.pred_state_length + z_size + x_size

        self.add_submodule(self.node_type + '/decoder/lstm_cell',
                           model_if_absent=nn.GRUCell(decoder_input_dims, self.hyperparams['dec_rnn_dim']))
        self.add_submodule(self.node_type + '/decoder/initial_h',
                           model_if_absent=nn.Linear(z_size + x_size, self.hyperparams['dec_rnn_dim']))
        self.add_submodule(self.node_type + '/decoder/initial_c',
                           model_if_absent=nn.Linear(z_size + x_size, self.hyperparams['dec_rnn_dim']))


        ###################
        #   Decoder GMM   #
        ###################
        self.add_submodule(self.node_type + '/decoder/proj_to_GMM_log_pis',
                           model_if_absent=nn.Linear(self.hyperparams['dec_rnn_dim'],
                                                     self.hyperparams['GMM_components']))
        self.add_submodule(self.node_type + '/decoder/proj_to_GMM_mus',
                           model_if_absent=nn.Linear(self.hyperparams['dec_rnn_dim'],
                                                     self.hyperparams['GMM_components'] * self.pred_state_length))
        self.add_submodule(self.node_type + '/decoder/proj_to_GMM_log_sigmas',
                           model_if_absent=nn.Linear(self.hyperparams['dec_rnn_dim'],
                                                     self.hyperparams['GMM_components'] * self.pred_state_length))
        self.add_submodule(self.node_type + '/decoder/proj_to_GMM_corrs',
                           model_if_absent=nn.Linear(self.hyperparams['dec_rnn_dim'],
                                                     self.hyperparams['GMM_components']))

        for name, module in self.node_modules.items():
            module.to(self.device)

    def create_new_scheduler(self, name, annealer, annealer_kws, creation_condition=True):
        value_scheduler = None
        rsetattr(self, name + '_scheduler', value_scheduler)
        if creation_condition:
            annealer_kws['device'] = self.device
            value_annealer = annealer(annealer_kws)
            rsetattr(self, name + '_annealer', value_annealer)

            # This is the value that we'll update on each call of
            # step_annealers().
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rsetattr(self, name, torch.tensor(value_annealer(0), device=self.device))
                dummy_optimizer = optim.Optimizer([rgetattr(self, name)],
                                                  {'lr': torch.tensor(value_annealer(0), device=self.device)})
                rsetattr(self, name + '_optimizer', dummy_optimizer)

            value_scheduler = CustomLR(dummy_optimizer,
                                       value_annealer)
            rsetattr(self, name + '_scheduler', value_scheduler)

        self.schedulers.append(value_scheduler)
        self.annealed_vars.append(name)

    def set_annealing_params(self):
        self.schedulers = list()
        self.annealed_vars = list()

        self.create_new_scheduler(name='kl_weight',
                                  annealer=sigmoid_anneal,
                                  annealer_kws={
                                      'start': self.hyperparams['kl_weight_start'],
                                      'finish': self.hyperparams['kl_weight'],
                                      'center_step': self.hyperparams['kl_crossover'],
                                      'steps_lo_to_hi': self.hyperparams['kl_crossover'] / self.hyperparams[
                                          'kl_sigmoid_divisor']
                                  },
                                  creation_condition=((np.abs(self.hyperparams['alpha'] - 1.0) < 1e-3)
                                                      and (not self.hyperparams['use_iwae'])))

        self.create_new_scheduler(name='dec_sample_model_prob',
                                  annealer=sigmoid_anneal,
                                  annealer_kws={
                                      'start': self.hyperparams['dec_sample_model_prob_start'],
                                      'finish': self.hyperparams['dec_sample_model_prob_final'],
                                      'center_step': self.hyperparams['dec_sample_model_prob_crossover'],
                                      'steps_lo_to_hi': self.hyperparams['dec_sample_model_prob_crossover'] /
                                                        self.hyperparams['dec_sample_model_prob_divisor']
                                  },
                                  creation_condition=self.hyperparams['sample_model_during_dec'])

        self.create_new_scheduler(name='latent.temp',
                                  annealer=exp_anneal,
                                  annealer_kws={
                                      'start': self.hyperparams['tau_init'],
                                      'finish': self.hyperparams['tau_final'],
                                      'rate': self.hyperparams['tau_decay_rate']
                                  })

        self.create_new_scheduler(name='latent.z_logit_clip',
                                  annealer=sigmoid_anneal,
                                  annealer_kws={
                                      'start': self.hyperparams['z_logit_clip_start'],
                                      'finish': self.hyperparams['z_logit_clip_final'],
                                      'center_step': self.hyperparams['z_logit_clip_crossover'],
                                      'steps_lo_to_hi': self.hyperparams['z_logit_clip_crossover'] / self.hyperparams[
                                          'z_logit_clip_divisor']
                                  },
                                  creation_condition=self.hyperparams['use_z_logit_clipping'])

        self.create_new_scheduler(name='warmup_dropout_keep',
                                  annealer=sigmoid_anneal,
                                  annealer_kws={
                                      'start': self.hyperparams['inf_warmup_start'],
                                      'finish': self.hyperparams['inf_warmup'],
                                      'center_step': self.hyperparams['inf_warmup_crossover'],
                                      'steps_lo_to_hi': self.hyperparams['inf_warmup_crossover'] / self.hyperparams[
                                          'inf_warmup_sigmoid_divisor']
                                  })

    def step_annealers(self):
        # This should manage all of the step-wise changed
        # parameters automatically.
        for idx, annealed_var in enumerate(self.annealed_vars):
            if rgetattr(self, annealed_var + '_scheduler') is not None:
                # First we step the scheduler.
                rgetattr(self, annealed_var + '_scheduler').step()

                # Then we set the annealed vars' value.
                rsetattr(self, annealed_var, rgetattr(self, annealed_var + '_optimizer').param_groups[0]['lr'])

        self.summarize_annealers()

    def summarize_annealers(self):
        if self.log_writer is not None:
            for annealed_var in self.annealed_vars:
                if rgetattr(self, annealed_var) is not None:
                    self.log_writer.add_scalar('%s/%s' % (str(self.node_type), annealed_var.replace('.', '/')),
                                               rgetattr(self, annealed_var), self.curr_iter)

    def obtain_encoded_tensor_dict(self,
                                   mode,
                                   timestep,
                                   timesteps_in_scene,
                                   inputs,
                                   inputs_st,
                                   labels,
                                   labels_st,
                                   first_history_indices,
                                   scene,
                                   node_scene_graph_batched):

        tensor_dict = dict()  # tensor_dict
        batch_size = inputs.shape[0]

        #########################################
        # Provide basic information to encoders #
        #########################################
        node_traj = inputs
        node_history = inputs[:, :timestep + 1]
        node_present_state = inputs[:, timestep]
        node_pos = inputs[:, timestep, 0:2]
        node_vel = inputs[:, timestep, 2:4]

        node_traj_st = inputs_st
        node_history_st = inputs_st[:, :timestep + 1]
        node_present_state_st = inputs_st[:, timestep]
        node_pos_st = inputs_st[:, timestep, 0:2]
        node_vel_st = inputs_st[:, timestep, 2:4]

        if self.hyperparams['incl_robot_node'] and scene.robot is not None:
            robot_traj_list = []
            for timestep_s in timesteps_in_scene:
                timestep_range = np.array([timestep_s - self.max_hl, timestep_s + node_traj.shape[1] - self.max_hl - 1])
                robot_traj_list.append(scene.robot.get(timestep_range, self.state[scene.robot.type.name]))
            robot_traj_np = np.array(robot_traj_list)

            # Make Robot State relative to node
            _, std = self.env.get_standardize_params(self.state[scene.robot.type.name], node_type=scene.robot.type)
            std[0:2] = 40
            rel_state = np.zeros_like(robot_traj_np)
            rel_state[..., :6] = node_traj[..., :6].cpu()
            robot_traj_np_st = self.env.standardize(robot_traj_np,
                                                        self.state[scene.robot.type.name],
                                                        node_type=scene.robot.type,
                                                        mean=rel_state,
                                                        std=std)
            robot_traj_st = torch.tensor(robot_traj_np_st).float().to(self.device)
            robot_present_state_st = robot_traj_st[:, timestep]
            robot_future_st = robot_traj_st[:, timestep+1:]

            tensor_dict['robot_present'] = robot_present_state_st
            tensor_dict['robot_future'] = robot_future_st

        ##################
        # Encode History #
        ##################
        tensor_dict['node_history_encoded'] = self.encode_node_history(mode,
                                                                       node_history_st,
                                                                       first_history_indices,
                                                                       timestep)

        ##################
        # Encode Present #
        ##################
        tensor_dict['node_present'] = node_present_state_st  # [bs, state_dim]

        ##################
        # Encode Future #
        ##################
        if mode != ModeKeys.PREDICT:
            tensor_dict['node_future'] = labels_st[:, timestep + 1:timestep + self.ph + 1]  # [bs, ph, state_dim]

        #######################################
        # Encode Joint Present (Robot + Node) #
        #######################################
        if self.warmup_dropout_keep < 0.5:
            if self.hyperparams['incl_robot_node'] and scene.robot is not None:
                tensor_dict['joint_present'] = torch.zeros_like(torch.cat([robot_present_state_st,
                                                                           labels_st[:, timestep]], dim=1))
            else:
                tensor_dict['joint_present'] = torch.zeros_like(labels_st[:, timestep])
        else:
            if self.hyperparams['incl_robot_node'] and scene.robot is not None:
                tensor_dict['joint_present'] = torch.cat([robot_present_state_st, labels_st[:, timestep]], dim=1)
            else:
                tensor_dict['joint_present'] = labels_st[:, timestep]

        ##############################
        # Encode Node Edges per Type #
        ##############################
        tensor_dict["node_edges_encoded"] = list()
        for edge_type in self.edge_types:
            connected_nodes_batched = list()
            edge_masks_batched = list()
            for i, (node, scene_graph) in enumerate(node_scene_graph_batched):
                # We get all nodes which are connected to the current node for the current timestep

                connected_nodes_batched.append(scene_graph.get_neighbors(node, edge_type[1]))

                if self.hyperparams['dynamic_edges'] == 'yes':
                    # We get the edge masks for the current node at the current timestep
                    edge_masks_for_node = scene_graph.get_edge_scaling(node)
                    edge_masks_batched.append(torch.tensor(edge_masks_for_node).float().to(self.device))

            # Encode edges for given edge type
            encoded_edges_type = self.encode_edge(mode,
                                                  node_history,
                                                  node_history_st,
                                                  edge_type,
                                                  connected_nodes_batched,
                                                  edge_masks_batched,
                                                  first_history_indices,
                                                  timestep,
                                                  timesteps_in_scene,
                                                  scene)
            tensor_dict["node_edges_encoded"].append(encoded_edges_type)  # List of [bs/nbs, enc_rnn_dim]
        #####################
        # Encode Node Edges #
        #####################
        tensor_dict["total_edge_influence"] = self.encode_total_edge_influence(mode,
                                                                               tensor_dict["node_edges_encoded"],
                                                                               tensor_dict["node_history_encoded"],
                                                                               batch_size)  # [bs/nbs, 4*enc_rnn_dim]
        #print(time.time() - t)
        ##############
        # Encode Map #
        ##############
        if mode == ModeKeys.TRAIN:
            rand_heading = (2 * np.random.rand(node_present_state.shape[0]) - 1) * 5 * np.pi / 180 # outside if because seeding
        else:
            rand_heading = 0.
        if self.hyperparams['use_map_encoding']:
            heading = node_present_state.cpu().numpy()[:, -1] + rand_heading
            node_pos_cpu = node_pos.cpu().numpy()
            if self.node_type == 'VEHICLE':
                node_pos_cpu = node_pos_cpu + 20 * np.array([np.cos(heading), np.sin(heading)]).T
            cropped_maps_np = get_cropped_maps_heading_exact(world_pts=node_pos_cpu,
                                            map=scene.map[self.node_type],
                                            context_size=self.hyperparams['map_context'],
                                                          heading=heading)
            cropped_maps_np = np.swapaxes(cropped_maps_np, -1, 1)
            cropped_maps = torch.from_numpy(cropped_maps_np).to(self.device)
            del cropped_maps_np
            encoded_map = self.node_modules[self.node_type + '/map_encoder'](cropped_maps)

            encoded_map = F.dropout(encoded_map, 0.5, training=(mode == ModeKeys.TRAIN))

            tensor_dict["encoded_maps"] = encoded_map

            if self.log_writer is not None and mode != ModeKeys.PREDICT:
                context_size = self.hyperparams['map_context']
                #cropped_maps = cropped_maps.clone()
                #cropped_maps[:, :, context_size // 2 - 3:context_size // 2 + 3, context_size // 2 - 3:context_size // 2 + 3] = 1.
                self.log_writer.add_images('%s/cropped_maps' % str(self.node_type),
                                           cropped_maps,
                                           self.curr_iter)

                img_pts = scene.map[self.node_type].to_map_points(node_pos_cpu)

                box_arr = np.empty((img_pts.shape[0], 4))
                box_arr[:, 0] = img_pts[:, 0] - context_size // 2
                box_arr[:, 1] = img_pts[:, 1] - context_size // 2
                box_arr[:, 2] = img_pts[:, 0] + context_size // 2
                box_arr[:, 3] = img_pts[:, 1] + context_size // 2

                self.log_writer.add_image_with_boxes('%s/cropped_locs' % str(self.node_type),
                                                     np.swapaxes(scene.map[self.node_type].fdata, -1, 0).astype(float),
                                                     box_arr,
                                                     self.curr_iter)

        ######################################
        # Concatenate Encoder Outputs into x #
        ######################################
        concat_list = list()
        if self.hyperparams['use_map_encoding']:
            concat_list.append(tensor_dict["encoded_maps"])  # [bs/nbs, map_enc_output_size]

        # Every node has an edge-influence encoder (which could just be zero).
        if self.warmup_dropout_keep < 0.5:
            concat_list.append(torch.zeros_like(tensor_dict["total_edge_influence"]))
        else:
            concat_list.append(tensor_dict["total_edge_influence"])  # [bs/nbs, 4*enc_rnn_dim]

        # Every node has a history encoder.
        if self.warmup_dropout_keep < 0.5:
            concat_list.append(torch.zeros_like(tensor_dict["node_history_encoded"]))
        else:
            concat_list.append(tensor_dict["node_history_encoded"])  # [bs/nbs, enc_rnn_dim_history]

        if self.hyperparams['incl_robot_node'] and scene.robot is not None:
            tensor_dict[scene.robot.type.name + "_robot_future_encoder"] = self.encode_robot_future(
                tensor_dict['robot_present'],
                tensor_dict['robot_future'],
                mode,
                scene.robot.type.name + '_robot')
            # [bs/nbs, 4*enc_rnn_dim_future]
            concat_list.append(tensor_dict[scene.robot.type.name + "_robot_future_encoder"])

        elif self.hyperparams['incl_robot_node']:
            # Four times because we're trying to mimic a bi-directional RNN's output (which is c and h from both ends).
            concat_list.append(
                torch.zeros([batch_size, 4 * self.hyperparams['enc_rnn_dim_future']], device=self.device))

        tensor_dict["x"] = torch.cat(concat_list, dim=1)

        if mode == ModeKeys.TRAIN or mode == ModeKeys.EVAL:
            tensor_dict[self.node_type + "_future_encoder"] = self.encode_node_future(tensor_dict['node_present'],
                                                                                      tensor_dict['node_future'],
                                                                                      mode,
                                                                                      self.node_type)

        return tensor_dict

    def encode_node_history(self, mode, node_traj, first_history_indices, timestep):
        outputs, _ = run_lstm_on_variable_length_seqs(self.node_modules[self.node_type + '/node_history_encoder'],
                                                      node_traj,
                                                      torch.ones_like(first_history_indices) * timestep,
                                                      first_history_indices,
                                                      self.hyperparams[
                                                          'maximum_history_length'] + 1)  # history + current

        outputs = F.dropout(outputs,
                            p=1. - self.hyperparams['rnn_kwargs']['dropout_keep_prob'],
                            training=(mode == ModeKeys.TRAIN))  # [bs, max_time, enc_rnn_dim]

        last_index_per_sequence = timestep - first_history_indices

        return outputs[torch.arange(first_history_indices.shape[0]), last_index_per_sequence]

    def encode_edge(self,
                    mode,
                    node_history,
                    node_history_st,
                    edge_type,
                    connected_nodes,
                    edge_masks,
                    first_history_indices,
                    timestep,
                    timesteps_in_scene,
                    scene):

        max_hl = self.hyperparams['maximum_history_length']

        edge_states_list = list()  # list of [#of neighbors, max_hl, state_dim]
        for i, timestep_in_scene in enumerate(timesteps_in_scene):  # Get neighbors for timestep in batch
            neighbor_states = list()
            for node in connected_nodes[i]:
                neighbor_state_np = node.get(np.array([timestep_in_scene - max_hl, timestep_in_scene]),
                                             self.state[node.type.name],
                                             padding=0.0)

                # Make State relative to node
                _, std = self.env.get_standardize_params(self.state[node.type.name], node_type=node.type)
                std[0:2] = self.env.attention_radius[edge_type]
                rel_state = np.zeros_like(neighbor_state_np)
                rel_state[:, :6] = node_history[i, -1, :6].cpu()
                neighbor_state_np_st = self.env.standardize(neighbor_state_np,
                                                            self.state[node.type.name],
                                                            node_type=node.type,
                                                            mean=rel_state,
                                                            std=std)

                neighbor_state = torch.tensor(neighbor_state_np_st).float().to(self.device)
                neighbor_states.append(neighbor_state)
            if len(neighbor_states) == 0:  # There are no neighbors for edge type # TODO necessary?
                neighbor_state_length = int(np.sum([len(entity_dims) for entity_dims in self.state[edge_type[1].name].values()]))
                edge_states_list.append(torch.zeros((1, max_hl + 1, neighbor_state_length), device=self.device))
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

        joint_history = torch.cat([combined_neighbors, node_history_st], dim=-1)

        outputs, _ = run_lstm_on_variable_length_seqs(self.node_modules[self.env.edge_type_str(edge_type) + '/edge_encoder'],
                                                      joint_history,
                                                      torch.ones_like(first_history_indices) * timestep,
                                                      first_history_indices,
                                                      self.hyperparams[
                                                          'maximum_history_length'] + 1)  # Add prediction timestep

        outputs = F.dropout(outputs,
                            p=1. - self.hyperparams['rnn_kwargs']['dropout_keep_prob'],
                            training=(mode == ModeKeys.TRAIN))  # [bs, max_time, enc_rnn_dim]

        last_index_per_sequence = timestep - first_history_indices
        ret = outputs[torch.arange(last_index_per_sequence.shape[0]), last_index_per_sequence]
        if self.hyperparams['dynamic_edges'] == 'yes':
            return ret * combined_edge_masks
        else:
            return ret

    def encode_total_edge_influence(self, mode, encoded_edges, node_history_encoder, batch_size):
        if self.hyperparams['edge_influence_combine_method'] == 'sum':
            stacked_encoded_edges = torch.stack(encoded_edges, dim=0)
            combined_edges = torch.sum(stacked_encoded_edges, dim=0)

        elif self.hyperparams['edge_influence_combine_method'] == 'mean':
            stacked_encoded_edges = torch.stack(encoded_edges, dim=0)
            combined_edges = torch.mean(stacked_encoded_edges, dim=0)

        elif self.hyperparams['edge_influence_combine_method'] == 'max':
            stacked_encoded_edges = torch.stack(encoded_edges, dim=0)
            combined_edges = torch.max(stacked_encoded_edges, dim=0)

        elif self.hyperparams['edge_influence_combine_method'] == 'bi-rnn':
            if len(encoded_edges) == 0:
                combined_edges = torch.zeros((batch_size, self.eie_output_dims), device=self.device)

            else:
                # axis=1 because then we get size [batch_size, max_time, depth]
                encoded_edges = torch.stack(encoded_edges, dim=1)

                _, state = self.node_modules[self.node_type + '/edge_influence_encoder'](encoded_edges)
                combined_edges = unpack_RNN_state(state)
                combined_edges = F.dropout(combined_edges,
                                           p=1. - self.hyperparams['rnn_kwargs']['dropout_keep_prob'],
                                           training=(mode == ModeKeys.TRAIN))

        elif self.hyperparams['edge_influence_combine_method'] == 'attention':
            # Used in Social Attention (https://arxiv.org/abs/1710.04689)
            if len(encoded_edges) == 0:
                combined_edges = torch.zeros((batch_size, self.eie_output_dims), device=self.device)

            else:
                # axis=1 because then we get size [batch_size, max_time, depth]
                encoded_edges = torch.stack(encoded_edges, dim=1)
                combined_edges, _ = self.node_modules[self.node_type + '/edge_influence_encoder'](encoded_edges,
                                                                                                  node_history_encoder)
                combined_edges = F.dropout(combined_edges,
                                           p=1. - self.hyperparams['rnn_kwargs']['dropout_keep_prob'],
                                           training=(mode == ModeKeys.TRAIN))

        return combined_edges

    def encode_node_future(self, node_present, node_future, mode, scope):
        initial_h_model = self.node_modules[self.node_type + '/node_future_encoder/initial_h']
        initial_c_model = self.node_modules[self.node_type + '/node_future_encoder/initial_c']

        # Here we're initializing the forward hidden states,
        # but zeroing the backward ones.
        initial_h = initial_h_model(node_present)
        initial_h = torch.stack([initial_h, torch.zeros_like(initial_h, device=self.device)], dim=0)

        initial_c = initial_c_model(node_present)
        initial_c = torch.stack([initial_c, torch.zeros_like(initial_c, device=self.device)], dim=0)

        initial_state = (initial_h, initial_c)

        _, state = self.node_modules[self.node_type + '/node_future_encoder'](node_future, initial_state)
        state = unpack_RNN_state(state)
        state = F.dropout(state,
                          p=1. - self.hyperparams['rnn_kwargs']['dropout_keep_prob'],
                          training=(mode == ModeKeys.TRAIN))

        return state

    def encode_robot_future(self, robot_present, robot_future, mode, scope):
        initial_h_model = self.node_modules['robot_future_encoder/initial_h']
        initial_c_model = self.node_modules['robot_future_encoder/initial_c']

        # Here we're initializing the forward hidden states,
        # but zeroing the backward ones.
        initial_h = initial_h_model(robot_present)
        initial_h = torch.stack([initial_h, torch.zeros_like(initial_h, device=self.device)], dim=0)

        initial_c = initial_c_model(robot_present)
        initial_c = torch.stack([initial_c, torch.zeros_like(initial_c, device=self.device)], dim=0)

        initial_state = (initial_h, initial_c)

        _, state = self.node_modules['robot_future_encoder'](robot_future, initial_state)
        state = unpack_RNN_state(state)
        state = F.dropout(state,
                          p=1. - self.hyperparams['rnn_kwargs']['dropout_keep_prob'],
                          training=(mode == ModeKeys.TRAIN))

        return state

    def q_z_xy(self, x, y, mode):
        xy = torch.cat([x, y], dim=1)

        if self.hyperparams['q_z_xy_MLP_dims'] is not None:
            dense = self.node_modules[self.node_type + '/q_z_xy']
            h = F.dropout(F.relu(dense(xy)),
                          p=1. - self.hyperparams['MLP_dropout_keep_prob'],
                          training=(mode == ModeKeys.TRAIN))

        else:
            h = xy

        to_latent = self.node_modules[self.node_type + '/hxy_to_z']
        return self.latent.dist_from_h(to_latent(h), mode)

    def p_z_x(self, x, mode):
        if self.hyperparams['p_z_x_MLP_dims'] is not None:
            dense = self.node_modules[self.node_type + '/p_z_x']
            h = F.dropout(F.relu(dense(x)),
                          p=1. - self.hyperparams['MLP_dropout_keep_prob'],
                          training=(mode == ModeKeys.TRAIN))

        else:
            h = x

        to_latent = self.node_modules[self.node_type + '/hx_to_z']
        return self.latent.dist_from_h(to_latent(h), mode)

    def project_to_GMM_params(self, tensor):
        log_pis = self.node_modules[self.node_type + '/decoder/proj_to_GMM_log_pis'](tensor)
        mus = self.node_modules[self.node_type + '/decoder/proj_to_GMM_mus'](tensor)
        log_sigmas = self.node_modules[self.node_type + '/decoder/proj_to_GMM_log_sigmas'](tensor)
        corrs = torch.tanh(self.node_modules[self.node_type + '/decoder/proj_to_GMM_corrs'](tensor))
        return log_pis, mus, log_sigmas, corrs

    def p_y_xz(self, x, z_stacked, tensor_dict, mode,
               num_predicted_timesteps, num_samples_z, num_samples_gmm=1, most_likely_gmm=False):
        ph = num_predicted_timesteps

        our_future = "node_future"
        robot_future = "robot_future"

        k = num_samples_z * num_samples_gmm
        GMM_c, pred_dim = self.hyperparams['GMM_components'], self.pred_state_length

        z = torch.reshape(z_stacked, (-1, self.latent.z_dim))
        zx = torch.cat([z, x.repeat(k, 1)], dim=1)

        cell = self.node_modules[self.node_type + '/decoder/lstm_cell']
        initial_h_model = self.node_modules[self.node_type + '/decoder/initial_h']

        initial_state = initial_h_model(zx)

        log_pis, mus, log_sigmas, corrs = [], [], [], []
        if mode in [ModeKeys.TRAIN, ModeKeys.EVAL]:
            state = initial_state
            if self.hyperparams['sample_model_during_dec'] and mode == ModeKeys.TRAIN:
                input_ = torch.cat([zx, tensor_dict['joint_present'].repeat(k, 1)], dim=1)
                for j in range(ph):
                    h_state = cell(input_, state)
                    log_pi_t, mu_t, log_sigma_t, corr_t = self.project_to_GMM_params(h_state)
                    y_t = GMM2D(log_pi_t, mu_t, log_sigma_t, corr_t, self.pred_state_length, self.device,
                                self.hyperparams['log_sigma_min'],
                                self.hyperparams['log_sigma_max']).sample()  # [k;bs, pred_dim]

                    # This is where we pick our output y_t or the true output
                    # our_future to pass into the next cell (we do this with
                    # probability self.dec_sample_model_prob and is only done
                    # during training).
                    mask = td.Bernoulli(probs=self.dec_sample_model_prob).sample((y_t.size()[0], 1))
                    y_t = mask * y_t + (1 - mask) * (tensor_dict[our_future][:, j, :].repeat(k, 1))

                    log_pis.append(log_pi_t)
                    mus.append(mu_t)
                    log_sigmas.append(log_sigma_t)
                    corrs.append(corr_t)

                    if self.hyperparams['incl_robot_node']:
                        dec_inputs = torch.cat([tensor_dict[robot_future][:, j, :].repeat(k, 1), y_t], dim=1)
                    else:
                        dec_inputs = y_t

                    input_ = torch.cat([zx, dec_inputs], dim=1)
                    state = h_state

                log_pis = torch.stack(log_pis, dim=1)
                mus = torch.stack(mus, dim=1)
                log_sigmas = torch.stack(log_sigmas, dim=1)
                corrs = torch.stack(corrs, dim=1)

            else:
                zx_with_time_dim = zx.unsqueeze(dim=1)  # [k;bs/nbs, 1, N*K + 2*enc_rnn_dim]
                zx_time_tiled = zx_with_time_dim.repeat(1, ph, 1)
                if self.hyperparams['incl_robot_node']:
                    dec_inputs = torch.cat([
                        tensor_dict["joint_present"].unsqueeze(dim=1),
                        torch.cat([tensor_dict[robot_future][:, :ph - 1, :], tensor_dict[our_future][:, :ph - 1, :]],
                                  dim=2)
                    ], dim=1)
                else:
                    dec_inputs = torch.cat([
                        tensor_dict["joint_present"].unsqueeze(dim=1),
                        tensor_dict[our_future][:, :ph - 1, :]
                    ], dim=1)

                outputs = list()
                for j in range(ph):
                    inputs = torch.cat([zx_time_tiled, dec_inputs.repeat(k, 1, 1)],
                                       dim=2)
                    h_state = cell(inputs[:, j, :], state)
                    outputs.append(h_state)
                    state = h_state

                outputs = torch.stack(outputs, dim=1)
                log_pis, mus, log_sigmas, corrs = self.project_to_GMM_params(outputs)

            if self.hyperparams['log_histograms'] and self.log_writer is not None:
                self.log_writer.add_histogram('%s/%s' % (str(self.node_type), 'GMM_log_pis'), log_pis, self.curr_iter)
                self.log_writer.add_histogram('%s/%s' % (str(self.node_type), 'GMM_mus'), mus, self.curr_iter)
                self.log_writer.add_histogram('%s/%s' % (str(self.node_type), 'GMM_log_sigmas'), log_sigmas,
                                              self.curr_iter)
                self.log_writer.add_histogram('%s/%s' % (str(self.node_type), 'GMM_corrs'), corrs, self.curr_iter)

        elif mode == ModeKeys.PREDICT:
            input_ = torch.cat([zx, tensor_dict["joint_present"].repeat(k, 1)], dim=1)
            state = initial_state

            log_pis, mus, log_sigmas, corrs, y = [], [], [], [], []
            for j in range(ph):
                h_state = cell(input_, state)
                log_pi_t, mu_t, log_sigma_t, corr_t = self.project_to_GMM_params(h_state)

                gmm = GMM2D(log_pi_t, mu_t, log_sigma_t, corr_t, self.pred_state_length, self.device,
                            self.hyperparams['log_sigma_min'],
                            self.hyperparams['log_sigma_max'])  # [k;bs, pred_dim]

                if most_likely_gmm:
                    y_t_list = []
                    for i in range(gmm.mus.shape[0]):
                        gmm_i = GMM2D(log_pi_t[i], mu_t[i], log_sigma_t[i], corr_t[i], self.pred_state_length,
                                      self.device,
                                      self.hyperparams['log_sigma_min'],
                                      self.hyperparams['log_sigma_max'])  # [k;bs, pred_dim]
                        x_min = gmm.mus[i, ..., 0].min()
                        x_max = gmm.mus[i, ..., 0].max()
                        y_min = gmm.mus[i, ..., 1].min()
                        y_max = gmm.mus[i, ..., 1].max()
                        x_min = x_min - 0.5 * torch.abs(x_min)
                        x_max = x_max + 0.5 * torch.abs(x_max)
                        y_min = y_min - 0.5 * torch.abs(y_min)
                        y_max = y_max + 0.5 * torch.abs(y_max)
                        search_grid = torch.stack(torch.meshgrid([torch.arange(x_min, x_max, 0.01),
                                                                  torch.arange(y_min, y_max, 0.01)]), dim=2
                                                  ).view(1, -1, 2).float().to(
                            self.device)

                        ll_score = gmm_i.log_prob(search_grid).squeeze()
                        y_t_list.append(search_grid[0, torch.argmax(ll_score, dim=0)])

                    y_t = torch.stack(y_t_list, dim=0)
                else:
                    y_t = gmm.sample()

                log_pis.append(log_pi_t)
                mus.append(mu_t)
                log_sigmas.append(log_sigma_t)
                corrs.append(corr_t)
                y.append(y_t)

                if self.hyperparams['incl_robot_node']:
                    dec_inputs = torch.cat([tensor_dict[robot_future][:, j, :].repeat(k, 1), y_t], dim=1)
                else:
                    dec_inputs = y_t

                input_ = torch.cat([zx, dec_inputs],dim=1)
                state = h_state

            log_pis = torch.stack(log_pis, dim=1)
            mus = torch.stack(mus, dim=1)
            log_sigmas = torch.stack(log_sigmas, dim=1)
            corrs = torch.stack(corrs, dim=1)
            sampled_future = torch.reshape(torch.stack(y, dim=1), (num_samples_z, num_samples_gmm, -1, ph, pred_dim))

        y_dist = GMM2D(torch.reshape(log_pis, [k, -1, ph, GMM_c]),
                       torch.reshape(mus, [k, -1, ph, GMM_c * pred_dim]),
                       torch.reshape(log_sigmas, [k, -1, ph, GMM_c * pred_dim]),
                       torch.reshape(corrs, [k, -1, ph, GMM_c]),
                       self.pred_state_length, self.device,
                       self.hyperparams['log_sigma_min'], self.hyperparams['log_sigma_max'])

        if mode == ModeKeys.PREDICT:
            return y_dist, sampled_future
        else:
            return y_dist

    def encoder(self, x, y, mode, num_samples=None):
        if mode == ModeKeys.TRAIN:
            sample_ct = self.hyperparams['k']
        elif mode == ModeKeys.EVAL:
            sample_ct = self.hyperparams['k_eval']
        elif mode == ModeKeys.PREDICT:
            sample_ct = num_samples
            if num_samples is None:
                raise ValueError("num_samples cannot be None with mode == PREDICT.")

        self.latent.q_dist = self.q_z_xy(x, y, mode)
        self.latent.p_dist = self.p_z_x(x, mode)

        z = self.latent.sample_q(sample_ct, mode)

        if mode == ModeKeys.TRAIN and self.hyperparams['kl_exact']:
            kl_obj = self.latent.kl_q_p(self.log_writer, '%s' % str(self.node_type), self.curr_iter)
            if self.log_writer is not None:
                self.log_writer.add_scalar('%s/%s' % (str(self.node_type), 'kl'), kl_obj, self.curr_iter)
        else:
            kl_obj = None

        return z, kl_obj

    def decoder(self, x, y, z, tensor_dict, mode, num_predicted_timesteps, num_samples):
        y_dist = self.p_y_xz(x, z, tensor_dict, mode, num_predicted_timesteps, num_samples)
        log_p_yt_xz = torch.clamp(y_dist.log_prob(y), max=self.hyperparams['log_p_yt_xz_max'])
        if self.hyperparams['log_histograms'] and self.log_writer is not None:
            self.log_writer.add_histogram('%s/%s' % (str(self.node_type), 'log_p_yt_xz'), log_p_yt_xz, self.curr_iter)

        log_p_y_xz = torch.sum(log_p_yt_xz, dim=2)
        return log_p_y_xz

    def mutual_inf_mc(self, x_dist):
        dist = x_dist.__class__
        H_y = dist(probs=x_dist.probs.mean(dim=0)).entropy()
        return (H_y - x_dist.entropy().mean(dim=0)).sum()

    def train_loss(self,
                   inputs,
                   inputs_st,
                   first_history_indices,
                   labels,
                   labels_st,
                   scene,
                   node_scene_graph_batched,
                   timestep,
                   timesteps_in_scene,
                   prediction_horizon):

        mode = ModeKeys.TRAIN

        tensor_dict = self.obtain_encoded_tensor_dict(mode,
                                                      timestep,
                                                      timesteps_in_scene,
                                                      inputs,
                                                      inputs_st,
                                                      labels,
                                                      labels_st,
                                                      first_history_indices,
                                                      scene,
                                                      node_scene_graph_batched)

        z, kl = self.encoder(tensor_dict["x"], tensor_dict[self.node_type + "_future_encoder"], mode)
        log_p_y_xz = self.decoder(tensor_dict["x"], tensor_dict["node_future"], z, tensor_dict, mode,
                                  prediction_horizon,
                                  self.hyperparams['k'])

        if np.abs(self.hyperparams['alpha'] - 1.0) < 1e-3 and not self.hyperparams['use_iwae']:
            log_p_y_xz_mean = torch.mean(log_p_y_xz, dim=0)  # [nbs]
            log_likelihood = torch.mean(log_p_y_xz_mean)

            mutual_inf_q = self.mutual_inf_mc(self.latent.q_dist)
            mutual_inf_p = self.mutual_inf_mc(self.latent.p_dist)

            ELBO = log_likelihood - self.kl_weight * kl + 1. * mutual_inf_p
            loss = -ELBO

            if self.hyperparams['log_histograms'] and self.log_writer is not None:
                self.log_writer.add_histogram('%s/%s' % (str(self.node_type), 'log_p_y_xz'),
                                              log_p_y_xz_mean,
                                              self.curr_iter)

        else:
            log_q_z_xy = self.latent.q_log_prob(z)  # [k, nbs]
            log_p_z_x = self.latent.p_log_prob(z)  # [k, nbs]
            a = self.hyperparams['alpha']
            log_pp_over_q = log_p_y_xz + log_p_z_x - log_q_z_xy
            log_likelihood = (torch.mean(torch.logsumexp(log_pp_over_q * (1. - a), dim=0))
                              - torch.log(self.hyperparams['k'])) / (1. - a)
            loss = -log_likelihood

        if self.log_writer is not None:
            self.log_writer.add_scalar('%s/%s' % (str(self.node_type), 'mutual_information_q'),
                                       mutual_inf_q,
                                       self.curr_iter)
            self.log_writer.add_scalar('%s/%s' % (str(self.node_type), 'mutual_information_p'),
                                       mutual_inf_p,
                                       self.curr_iter)
            self.log_writer.add_scalar('%s/%s' % (str(self.node_type), 'log_likelihood'),
                                       log_likelihood,
                                       self.curr_iter)
            self.log_writer.add_scalar('%s/%s' % (str(self.node_type), 'loss'),
                                       loss,
                                       self.curr_iter)
            if self.hyperparams['log_histograms']:
                self.latent.summarize_for_tensorboard(self.log_writer, str(self.node_type), self.curr_iter)

        return loss

    def eval_loss(self,
                  inputs,
                  inputs_st,
                  first_history_indices,
                  labels,
                  labels_st,
                  scene,
                  node_scene_graph_batched,
                  timestep,
                  timesteps_in_scene,
                  prediction_horizon,
                  compute_naive=True,
                  compute_exact=True,
                  compute_sample=True):
        mode = ModeKeys.EVAL

        tensor_dict = self.obtain_encoded_tensor_dict(mode,
                                                      timestep,
                                                      timesteps_in_scene,
                                                      inputs,
                                                      inputs_st,
                                                      labels,
                                                      labels_st,
                                                      first_history_indices,
                                                      scene,
                                                      node_scene_graph_batched)

        ### Importance sampled NLL estimate
        z, _ = self.encoder(tensor_dict["x"], tensor_dict[self.node_type + "_future_encoder"],
                            mode)  # [k_eval, nbs, N*K]
        log_p_y_xz = self.decoder(tensor_dict["x"], tensor_dict['node_future'], z, tensor_dict, mode,
                                  prediction_horizon,
                                  self.hyperparams['k_eval'])  # [k_eval, nbs]
        log_q_z_xy = self.latent.q_log_prob(z)  # [k_eval, nbs]
        log_p_z_x = self.latent.p_log_prob(z)  # [k_eval, nbs]
        log_likelihood = torch.mean(torch.logsumexp(log_p_y_xz + log_p_z_x - log_q_z_xy, dim=0)) - \
                         torch.log(torch.tensor(self.hyperparams['k_eval'], dtype=torch.float, device=self.device))
        nll_q_is = -log_likelihood

        ### Naive sampled NLL estimate
        nll_p = torch.tensor(np.nan)
        if compute_naive:
            z = self.latent.sample_p(self.hyperparams['k_eval'], mode)
            log_p_y_xz = self.decoder(tensor_dict["x"], tensor_dict['node_future'], z, tensor_dict, mode,
                                      prediction_horizon,
                                      self.hyperparams['k_eval'])
            log_likelihood_p = torch.mean(torch.logsumexp(log_p_y_xz, dim=0)) - \
                               torch.log(
                                   torch.tensor(self.hyperparams['k_eval'], dtype=torch.float, device=self.device))
            nll_p = -log_likelihood_p

        ### Exact NLL
        nll_exact = torch.tensor(np.nan)
        if compute_exact:
            K, N = self.hyperparams['K'], self.hyperparams['N']
            if K ** N < 50:
                nbs = tensor_dict["x"].size()[0]
                z_raw = torch.from_numpy(
                    DiscreteLatent.all_one_hot_combinations(N, K).astype(np.float32)
                ).to(self.device).repeat(1, nbs)  # [K**N, nbs*N*K]

                z = torch.reshape(z_raw, (K ** N, -1, N * K))  # [K**N, nbs, N*K]
                log_p_y_xz = self.decoder(tensor_dict["x"], tensor_dict['node_future'], z, tensor_dict, mode,
                                          prediction_horizon,
                                          K ** N)  # [K**N, nbs]
                log_p_z_x = self.latent.p_log_prob(z)  # [K**N, nbs]
                exact_log_likelihood = torch.mean(torch.logsumexp(log_p_y_xz + log_p_z_x, dim=0))

                nll_exact = -exact_log_likelihood

        nll_sampled = torch.tensor(np.nan)
        if compute_sample:
            z = self.latent.sample_p(self.hyperparams['k_eval'], mode)
            y_dist, _ = self.p_y_xz(tensor_dict["x"], z, tensor_dict, ModeKeys.PREDICT, prediction_horizon,
                                    self.hyperparams['k_eval'])
            log_p_yt_xz = torch.clamp(y_dist.log_prob(tensor_dict['node_future']),
                                      max=self.hyperparams['log_p_yt_xz_max'])
            log_p_y_xz = torch.sum(log_p_yt_xz, dim=2)
            log_p_y_xz_mean = torch.mean(log_p_y_xz, dim=0)  # [nbs]
            log_likelihood = torch.mean(log_p_y_xz_mean)
            nll_sampled = -log_likelihood

        if self.log_writer is not None:
            self.log_writer.add_scalar('%s/%s' % (str(self.node_type), 'log_likelihood_eval'),
                                       log_likelihood,
                                       self.curr_iter)

        return nll_q_is, nll_p, nll_exact, nll_sampled

    def predict(self,
                inputs,
                inputs_st,
                labels,
                labels_st,
                first_history_indices,
                scene,
                node_scene_graph_batched,
                timestep,
                timesteps_in_scene,
                prediction_horizon,
                num_samples_z,
                num_samples_gmm,
                most_likely_z=False,
                most_likely_gmm=False,
                all_z=False):
        mode = ModeKeys.PREDICT

        tensor_dict = self.obtain_encoded_tensor_dict(mode,
                                                      timestep,
                                                      timesteps_in_scene,
                                                      inputs,
                                                      inputs_st,
                                                      labels,
                                                      labels_st,
                                                      first_history_indices,
                                                      scene,
                                                      node_scene_graph_batched)

        self.latent.p_dist = self.p_z_x(tensor_dict["x"], mode)
        z, num_samples_z = self.latent.sample_p(num_samples_z,
                                                mode,
                                                num_samples_gmm=num_samples_gmm,
                                                most_likely=most_likely_z,
                                                all_z=all_z)

        y_dist, our_sampled_future = self.p_y_xz(tensor_dict["x"], z, tensor_dict, mode,
                                                 prediction_horizon,
                                                 num_samples_z,
                                                 num_samples_gmm,
                                                 most_likely_gmm)  # y_dist.mean is [k, bs, ph*state_dim]

        return our_sampled_future
