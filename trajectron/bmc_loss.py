import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch.distributions import MultivariateNormal as MVN

def contrastive_three_modes_loss(features, scores, temp=0.1, base_temperature=0.07):
    device = (torch.device('cuda') if features.is_cuda
              else torch.device('cpu'))
    batch_size = features.shape[0]
    scores = scores.contiguous().view(-1, 1)
    mask_positives = (torch.abs(scores.sub(scores.T)) < 0.1).float().to(device)
    mask_negatives = (torch.abs(scores.sub(scores.T)) > 2.0).float().to(device)
    mask_neutral = mask_positives + mask_negatives

    anchor_dot_contrast = torch.div(torch.matmul(features, features.T), temp)
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    logits_mask = torch.scatter(
        torch.ones_like(mask_positives), 1,
        torch.arange(batch_size).view(-1, 1).to(device), 0) * mask_neutral
    mask_positives = mask_positives * logits_mask
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-20)
    mean_log_prob_pos = (mask_positives * log_prob).sum(1) / (mask_positives.sum(1) + 1e-20)

    loss = - (temp / base_temperature) * mean_log_prob_pos
    loss = loss.view(1, batch_size).mean()
    return loss 

def bmc_loss_md(y, labels, noise_var):
    """Compute the Multidimensional Balanced MSE Loss (BMC) between `pred` and the ground truth `targets`.
    Args:
      pred: A float tensor of size [batch, d].
      target: A float tensor of size [batch, d].
      noise_var: A float number or tensor.
    Returns:
      loss: A float tensor. Balanced MSE Loss.
    """
    # y has shape (bs, 12 ,2)
    # labels has shape (bs, 12, 2)
    pred = torch.flatten(y, start_dim=1)
    # target = torch.stack([labels for i in range(20)], dim=1)  # (bs, 20, 12, 2)
    target = torch.flatten(labels, start_dim=1)

    device = (torch.device('cuda') if y.is_cuda
            else torch.device('cpu'))
    I = torch.eye(pred.shape[-1]).to(device)
    logits = MVN(pred.unsqueeze(1), noise_var*I).log_prob(target.unsqueeze(0)).to(device)  # logit size: [batch, batch]
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0]).to(device))     # contrastive-like loss
    loss = loss * (2 * noise_var).detach()  # optional: restore the loss scale, 'detach' when noise is learnable 
    
    return loss

class BMCLoss(_Loss):
    def __init__(self, init_noise_sigma, device):
        super(BMCLoss, self).__init__()
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma, device=device, dtype=torch.float))

    def forward(self, pred, target):
        noise_var = self.noise_sigma ** 2
        loss = bmc_loss_md(pred, target, noise_var)
        return loss

class ContrastiveLoss(_Loss):
    def __init__(self, init_temp, device):
        super(ContrastiveLoss, self).__init__()
        self.temp = torch.nn.Parameter(torch.tensor(init_temp, device=device, dtype=torch.float))

    def forward(self, z, scores):
        loss = contrastive_three_modes_loss(z, scores, temp=self.temp)
        return loss
