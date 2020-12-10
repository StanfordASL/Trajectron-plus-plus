import torch
import torch.nn as nn
from model.dynamics import Dynamic
from utils import block_diag
from model.components import GMM2D


class Unicycle(Dynamic):
    def init_constants(self):
        self.F_s = torch.eye(4, device=self.device, dtype=torch.float32)
        self.F_s[0:2, 2:] = torch.eye(2, device=self.device, dtype=torch.float32) * self.dt
        self.F_s_t = self.F_s.transpose(-2, -1)

    def create_graph(self, xz_size):
        model_if_absent = nn.Linear(xz_size + 1, 1)
        self.p0_model = self.model_registrar.get_model(f"{self.node_type}/unicycle_initializer", model_if_absent)

    def dynamic(self, x, u):
        r"""
        TODO: Boris: Add docstring
        :param x:
        :param u:
        :return:
        """
        x_p = x[0]
        y_p = x[1]
        phi = x[2]
        v = x[3]
        dphi = u[0]
        a = u[1]

        mask = torch.abs(dphi) <= 1e-2
        dphi = ~mask * dphi + (mask) * 1

        phi_p_omega_dt = phi + dphi * self.dt
        dsin_domega = (torch.sin(phi_p_omega_dt) - torch.sin(phi)) / dphi
        dcos_domega = (torch.cos(phi_p_omega_dt) - torch.cos(phi)) / dphi

        d1 = torch.stack([(x_p
                           + (a / dphi) * dcos_domega
                           + v * dsin_domega
                           + (a / dphi) * torch.sin(phi_p_omega_dt) * self.dt),
                          (y_p
                           - v * dcos_domega
                           + (a / dphi) * dsin_domega
                           - (a / dphi) * torch.cos(phi_p_omega_dt) * self.dt),
                          phi + dphi * self.dt,
                          v + a * self.dt], dim=0)
        d2 = torch.stack([x_p + v * torch.cos(phi) * self.dt + (a / 2) * torch.cos(phi) * self.dt ** 2,
                          y_p + v * torch.sin(phi) * self.dt + (a / 2) * torch.sin(phi) * self.dt ** 2,
                          phi * torch.ones_like(a),
                          v + a * self.dt], dim=0)
        return torch.where(~mask, d1, d2)

    def integrate_samples(self, control_samples, x=None):
        r"""
        TODO: Boris: Add docstring
        :param x:
        :param u:
        :return:
        """
        ph = control_samples.shape[-2]
        p_0 = self.initial_conditions['pos'].unsqueeze(1)
        v_0 = self.initial_conditions['vel'].unsqueeze(1)

        # In case the input is batched because of the robot in online use we repeat this to match the batch size of x.
        if p_0.size()[0] != x.size()[0]:
            p_0 = p_0.repeat(x.size()[0], 1, 1)
            v_0 = v_0.repeat(x.size()[0], 1, 1)

        phi_0 = torch.atan2(v_0[..., 1], v_0[..., 0])

        phi_0 = phi_0 + torch.tanh(self.p0_model(torch.cat((x, phi_0), dim=-1)))

        u = torch.stack([control_samples[..., 0], control_samples[..., 1]], dim=0)
        x = torch.stack([p_0[..., 0], p_0[..., 1], phi_0, torch.norm(v_0, dim=-1)], dim = 0).squeeze(dim=-1)

        mus_list = []
        for t in range(ph):
            x = self.dynamic(x, u[..., t])
            mus_list.append(torch.stack((x[0], x[1]), dim=-1))

        pos_mus = torch.stack(mus_list, dim=2)
        return pos_mus

    def compute_control_jacobian(self, sample_batch_dim, components, x, u):
        r"""
        TODO: Boris: Add docstring
        :param x:
        :param u:
        :return:
        """
        F = torch.zeros(sample_batch_dim + [components, 4, 2],
                        device=self.device,
                        dtype=torch.float32)

        phi = x[2]
        v = x[3]
        dphi = u[0]
        a = u[1]

        mask = torch.abs(dphi) <= 1e-2
        dphi = ~mask * dphi + (mask) * 1

        phi_p_omega_dt = phi + dphi * self.dt
        dsin_domega = (torch.sin(phi_p_omega_dt) - torch.sin(phi)) / dphi
        dcos_domega = (torch.cos(phi_p_omega_dt) - torch.cos(phi)) / dphi

        F[..., 0, 0] = ((v / dphi) * torch.cos(phi_p_omega_dt) * self.dt
                        - (v / dphi) * dsin_domega
                        - (2 * a / dphi ** 2) * torch.sin(phi_p_omega_dt) * self.dt
                        - (2 * a / dphi ** 2) * dcos_domega
                        + (a / dphi) * torch.cos(phi_p_omega_dt) * self.dt ** 2)
        F[..., 0, 1] = (1 / dphi) * dcos_domega + (1 / dphi) * torch.sin(phi_p_omega_dt) * self.dt

        F[..., 1, 0] = ((v / dphi) * dcos_domega
                        - (2 * a / dphi ** 2) * dsin_domega
                        + (2 * a / dphi ** 2) * torch.cos(phi_p_omega_dt) * self.dt
                        + (v / dphi) * torch.sin(phi_p_omega_dt) * self.dt
                        + (a / dphi) * torch.sin(phi_p_omega_dt) * self.dt ** 2)
        F[..., 1, 1] = (1 / dphi) * dsin_domega - (1 / dphi) * torch.cos(phi_p_omega_dt) * self.dt

        F[..., 2, 0] = self.dt

        F[..., 3, 1] = self.dt

        F_sm = torch.zeros(sample_batch_dim + [components, 4, 2],
                           device=self.device,
                           dtype=torch.float32)

        F_sm[..., 0, 1] = (torch.cos(phi) * self.dt ** 2) / 2

        F_sm[..., 1, 1] = (torch.sin(phi) * self.dt ** 2) / 2

        F_sm[..., 3, 1] = self.dt

        return torch.where(~mask.unsqueeze(-1).unsqueeze(-1), F, F_sm)

    def compute_jacobian(self, sample_batch_dim, components, x, u):
        r"""
        TODO: Boris: Add docstring
        :param x:
        :param u:
        :return:
        """
        one = torch.tensor(1)
        F = torch.zeros(sample_batch_dim + [components, 4, 4],
                        device=self.device,
                        dtype=torch.float32)

        phi = x[2]
        v = x[3]
        dphi = u[0]
        a = u[1]

        mask = torch.abs(dphi) <= 1e-2
        dphi = ~mask * dphi + (mask) * 1

        phi_p_omega_dt = phi + dphi * self.dt
        dsin_domega = (torch.sin(phi_p_omega_dt) - torch.sin(phi)) / dphi
        dcos_domega = (torch.cos(phi_p_omega_dt) - torch.cos(phi)) / dphi

        F[..., 0, 0] = one
        F[..., 1, 1] = one
        F[..., 2, 2] = one
        F[..., 3, 3] = one

        F[..., 0, 2] = v * dcos_domega - (a / dphi) * dsin_domega + (a / dphi) * torch.cos(phi_p_omega_dt) * self.dt
        F[..., 0, 3] = dsin_domega

        F[..., 1, 2] = v * dsin_domega + (a / dphi) * dcos_domega + (a / dphi) * torch.sin(phi_p_omega_dt) * self.dt
        F[..., 1, 3] = -dcos_domega

        F_sm = torch.zeros(sample_batch_dim + [components, 4, 4],
                           device=self.device,
                           dtype=torch.float32)

        F_sm[..., 0, 0] = one
        F_sm[..., 1, 1] = one
        F_sm[..., 2, 2] = one
        F_sm[..., 3, 3] = one

        F_sm[..., 0, 2] = -v * torch.sin(phi) * self.dt - (a * torch.sin(phi) * self.dt ** 2) / 2
        F_sm[..., 0, 3] = torch.cos(phi) * self.dt

        F_sm[..., 1, 2] = v * torch.cos(phi) * self.dt + (a * torch.cos(phi) * self.dt ** 2) / 2
        F_sm[..., 1, 3] = torch.sin(phi) * self.dt

        return torch.where(~mask.unsqueeze(-1).unsqueeze(-1), F, F_sm)

    def integrate_distribution(self, control_dist_dphi_a, x):
        r"""
        TODO: Boris: Add docstring
        :param x:
        :param u:
        :return:
        """
        sample_batch_dim = list(control_dist_dphi_a.mus.shape[0:2])
        ph = control_dist_dphi_a.mus.shape[-3]
        p_0 = self.initial_conditions['pos'].unsqueeze(1)
        v_0 = self.initial_conditions['vel'].unsqueeze(1)

        # In case the input is batched because of the robot in online use we repeat this to match the batch size of x.
        if p_0.size()[0] != x.size()[0]:
            p_0 = p_0.repeat(x.size()[0], 1, 1)
            v_0 = v_0.repeat(x.size()[0], 1, 1)

        phi_0 = torch.atan2(v_0[..., 1], v_0[..., 0])

        phi_0 = phi_0 + torch.tanh(self.p0_model(torch.cat((x, phi_0), dim=-1)))

        dist_sigma_matrix = control_dist_dphi_a.get_covariance_matrix()
        pos_dist_sigma_matrix_t = torch.zeros(sample_batch_dim + [control_dist_dphi_a.components, 4, 4],
                                              device=self.device)

        u = torch.stack([control_dist_dphi_a.mus[..., 0], control_dist_dphi_a.mus[..., 1]], dim=0)
        x = torch.stack([p_0[..., 0], p_0[..., 1], phi_0, torch.norm(v_0, dim=-1)], dim=0)

        pos_dist_sigma_matrix_list = []
        mus_list = []
        for t in range(ph):
            F_t = self.compute_jacobian(sample_batch_dim, control_dist_dphi_a.components, x, u[:, :, :, t])
            G_t = self.compute_control_jacobian(sample_batch_dim, control_dist_dphi_a.components, x, u[:, :, :, t])
            dist_sigma_matrix_t = dist_sigma_matrix[:, :, t]
            pos_dist_sigma_matrix_t = (F_t.matmul(pos_dist_sigma_matrix_t.matmul(F_t.transpose(-2, -1)))
                                       + G_t.matmul(dist_sigma_matrix_t.matmul(G_t.transpose(-2, -1))))
            pos_dist_sigma_matrix_list.append(pos_dist_sigma_matrix_t[..., :2, :2])

            x = self.dynamic(x, u[:, :, :, t])
            mus_list.append(torch.stack((x[0], x[1]), dim=-1))

        pos_dist_sigma_matrix = torch.stack(pos_dist_sigma_matrix_list, dim=2)
        pos_mus = torch.stack(mus_list, dim=2)
        return GMM2D.from_log_pis_mus_cov_mats(control_dist_dphi_a.log_pis, pos_mus, pos_dist_sigma_matrix)
