import torch
import torch.distributions as td
import numpy as np
from model.model_utils import ModeKeys


class DiscreteLatent(object):
    def __init__(self, hyperparams, device):
        self.hyperparams = hyperparams
        self.z_dim = hyperparams['N'] * hyperparams['K']
        self.N = hyperparams['N']
        self.K = hyperparams['K']
        self.kl_min = hyperparams['kl_min']
        self.device = device
        self.temp = None  # filled in by MultimodalGenerativeCVAE.set_annealing_params
        self.z_logit_clip = None  # filled in by MultimodalGenerativeCVAE.set_annealing_params
        self.p_dist = None  # filled in by MultimodalGenerativeCVAE.encoder
        self.q_dist = None  # filled in by MultimodalGenerativeCVAE.encoder

    def dist_from_h(self, h, mode):
        logits_separated = torch.reshape(h, (-1, self.N, self.K))
        logits_separated_mean_zero = logits_separated - torch.mean(logits_separated, dim=-1, keepdim=True)
        if self.z_logit_clip is not None and mode == ModeKeys.TRAIN:
            c = self.z_logit_clip
            logits = torch.clamp(logits_separated_mean_zero, min=-c, max=c)
        else:
            logits = logits_separated_mean_zero

        return td.OneHotCategorical(logits=logits)

    def sample_q(self, num_samples, mode):
        bs = self.p_dist.probs.size()[0]
        num_components = self.N * self.K
        z_NK = torch.from_numpy(self.all_one_hot_combinations(self.N, self.K)).float().to(self.device).repeat(num_samples, bs)
        return torch.reshape(z_NK, (num_samples * num_components, -1, self.z_dim))

    def sample_p(self, num_samples, mode, most_likely_z=False, full_dist=True, all_z_sep=False):
        num_components = 1
        if full_dist:
            bs = self.p_dist.probs.size()[0]
            z_NK = torch.from_numpy(self.all_one_hot_combinations(self.N, self.K)).float().to(self.device).repeat(num_samples, bs)
            num_components = self.K ** self.N
            k = num_samples * num_components
        elif all_z_sep:
            bs = self.p_dist.probs.size()[0]
            z_NK = torch.from_numpy(self.all_one_hot_combinations(self.N, self.K)).float().to(self.device).repeat(1, bs)
            k = self.K ** self.N
            num_samples = k
        elif most_likely_z:
            # Sampling the most likely z from p(z|x).
            eye_mat = torch.eye(self.p_dist.event_shape[-1], device=self.device)
            argmax_idxs = torch.argmax(self.p_dist.probs, dim=2)
            z_NK = torch.unsqueeze(eye_mat[argmax_idxs], dim=0).expand(num_samples, -1, -1, -1)
            k = num_samples
        else:
            z_NK = self.p_dist.sample((num_samples,))
            k = num_samples

        if mode == ModeKeys.PREDICT:
            return torch.reshape(z_NK, (k, -1, self.N * self.K)), num_samples, num_components
        else:
            return torch.reshape(z_NK, (k, -1, self.N * self.K))

    def kl_q_p(self, log_writer=None, prefix=None, curr_iter=None):
        kl_separated = td.kl_divergence(self.q_dist, self.p_dist)
        if len(kl_separated.size()) < 2:
            kl_separated = torch.unsqueeze(kl_separated, dim=0)

        kl_minibatch = torch.mean(kl_separated, dim=0, keepdim=True)

        if log_writer is not None:
            log_writer.add_scalar(prefix + '/true_kl', torch.sum(kl_minibatch), curr_iter)

        if self.kl_min > 0:
            kl_lower_bounded = torch.clamp(kl_minibatch, min=self.kl_min)
            kl = torch.sum(kl_lower_bounded)
        else:
            kl = torch.sum(kl_minibatch)

        return kl

    def q_log_prob(self, z):
        k = z.size()[0]
        z_NK = torch.reshape(z, [k, -1, self.N, self.K])
        return torch.sum(self.q_dist.log_prob(z_NK), dim=2)

    def p_log_prob(self, z):
        k = z.size()[0]
        z_NK = torch.reshape(z, [k, -1, self.N, self.K])
        return torch.sum(self.p_dist.log_prob(z_NK), dim=2)

    def get_p_dist_probs(self):
        return self.p_dist.probs

    @staticmethod
    def all_one_hot_combinations(N, K):
        return np.eye(K).take(np.reshape(np.indices([K] * N), [N, -1]).T, axis=0).reshape(-1, N * K)  # [K**N, N*K]

    def summarize_for_tensorboard(self, log_writer, prefix, curr_iter):
        log_writer.add_histogram(prefix + "/latent/p_z_x", self.p_dist.probs, curr_iter)
        log_writer.add_histogram(prefix + "/latent/q_z_xy", self.q_dist.probs, curr_iter)
        log_writer.add_histogram(prefix + "/latent/p_z_x_logits", self.p_dist.logits, curr_iter)
        log_writer.add_histogram(prefix + "/latent/q_z_xy_logits", self.q_dist.logits, curr_iter)
        if self.z_dim <= 9:
            for i in range(self.N):
                for j in range(self.K):
                    log_writer.add_histogram(prefix + "/latent/q_z_xy_logit{0}{1}".format(i, j),
                                             self.q_dist.logits[:, i, j],
                                             curr_iter)
