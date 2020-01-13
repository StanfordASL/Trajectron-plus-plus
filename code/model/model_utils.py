import torch
import torch.nn.utils.rnn as rnn
from enum import Enum
import functools
import numpy as np
import math
from scipy.ndimage import interpolation


class ModeKeys(Enum):
    TRAIN = 1
    EVAL = 2
    PREDICT = 3

def cyclical_lr(stepsize, min_lr=3e-4, max_lr=3e-3, decay=1.):

    # Lambda function to calculate the LR
    lr_lambda = lambda it: min_lr + (max_lr - min_lr) * relative(it, stepsize) * decay**it

    # Additional function to see where on the cycle we are
    def relative(it, stepsize):
        cycle = math.floor(1 + it / (2 * stepsize))
        x = abs(it / stepsize - 2 * cycle + 1)
        return max(0, (1 - x))

    return lr_lambda


def tile(a, dim, n_tile, device='cpu'):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(device)
    return torch.index_select(a, dim, order_index)


def to_one_hot(labels, n_labels, device):
    return torch.eye(n_labels, device=device)[labels]


def exp_anneal(anneal_kws):
    device = anneal_kws['device']
    start = torch.tensor(anneal_kws['start'], device=device)
    finish = torch.tensor(anneal_kws['finish'], device=device)
    rate = torch.tensor(anneal_kws['rate'], device=device)
    return lambda step: finish - (finish - start)*torch.pow(rate, torch.tensor(step, dtype=torch.float, device=device))


def sigmoid_anneal(anneal_kws):
    device = anneal_kws['device']
    start = torch.tensor(anneal_kws['start'], device=device)
    finish = torch.tensor(anneal_kws['finish'], device=device)
    center_step = torch.tensor(anneal_kws['center_step'], device=device, dtype=torch.float)
    steps_lo_to_hi = torch.tensor(anneal_kws['steps_lo_to_hi'], device=device, dtype=torch.float)
    return lambda step: start + (finish - start)*torch.sigmoid((torch.tensor(float(step), device=device) - center_step) * (1./steps_lo_to_hi))


class CustomLR(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, lr_lambda, last_epoch=-1):
        super(CustomLR, self).__init__(optimizer, lr_lambda, last_epoch)

    def get_lr(self):
        return [lmbda(self.last_epoch)
                for lmbda, base_lr in zip(self.lr_lambdas, self.base_lrs)]


def run_lstm_on_variable_length_seqs(lstm_module, original_seqs, break_indices, lower_indices, total_length):
    # This is done so that we can just pass in self.prediction_timesteps
    # (which we want to INCLUDE, so this will exclude the next timestep).
    inclusive_break_indices = break_indices + 1

    pad_list = list()
    for i, seq_len in enumerate(inclusive_break_indices):
        pad_list.append(original_seqs[i, lower_indices[i]:seq_len])

    packed_seqs = rnn.pack_sequence(pad_list, enforce_sorted=False)
    packed_output, (h_n, c_n) = lstm_module(packed_seqs)
    output, _ = rnn.pad_packed_sequence(packed_output,
                                        batch_first=True,
                                        total_length=total_length)

    return output, (h_n, c_n)


def extract_subtensor_per_batch_element(tensor, indices):
    batch_idxs = torch.arange(start=0, end=len(indices))

    batch_idxs = batch_idxs[~torch.isnan(indices)]
    indices = indices[~torch.isnan(indices)]
    if indices.size == 0:
        return None
    else:
        indices = indices.long()
    if tensor.is_cuda:
        batch_idxs = batch_idxs.to(tensor.get_device())
        indices = indices.to(tensor.get_device())
    return tensor[batch_idxs, indices]


def unpack_RNN_state(state_tuple):
    # PyTorch returned LSTM states have 3 dims:
    # (num_layers * num_directions, batch, hidden_size)

    state = torch.cat(state_tuple, dim=0).permute(1, 0, 2)
    # Now state is (batch, 2 * num_layers * num_directions, hidden_size)

    state_size = state.size()
    return torch.reshape(state, (-1, state_size[1] * state_size[2]))


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


# using wonder's beautiful simplification:
# https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427
def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


def get_cropped_maps(world_pts, map, context_size=50):
    """world_pts: N x 2 array of positions relative to the world."""
    expanded_obs_img = np.full((map.data.shape[0] + context_size, map.data.shape[1] + context_size, map.data.shape[2]), False, dtype=np.float32)
    expanded_obs_img[context_size//2:-context_size//2, context_size//2:-context_size//2] = map.fdata.astype(np.float32)
    img_pts = context_size//2 + np.round(map.to_map_points(world_pts)).astype(int)
    return np.stack([expanded_obs_img[img_pts[i, 0] - context_size//2 : img_pts[i, 0] + context_size//2,
                                      img_pts[i, 1] - context_size//2 : img_pts[i, 1] + context_size//2]
                      for i in range(world_pts.shape[0])], axis=0)

def get_cropped_maps_heading(world_pts, map, context_size=50, heading=None):
    """world_pts: N x 2 array of positions relative to the world."""
    rotations = np.round((heading) / (np.pi / 2)).astype(int)

    expanded_obs_img = np.full((map.data.shape[0] + context_size, map.data.shape[1] + context_size, map.data.shape[2]),
                               False, dtype=np.float32)
    expanded_obs_img[context_size // 2:-context_size // 2, context_size // 2:-context_size // 2] = map.fdata.astype(
        np.float32)
    img_pts = context_size // 2 + np.round(map.to_map_points(world_pts)).astype(int)
    map_h =  np.stack([expanded_obs_img[img_pts[i, 0] - context_size // 2: img_pts[i, 0] + context_size // 2,
                     img_pts[i, 1] - context_size // 2: img_pts[i, 1] + context_size // 2]
                     for i in range(world_pts.shape[0])], axis=0)

    map_h[rotations == 1] = np.rot90(map_h[rotations == 1], -1, axes=(1, 2))
    map_h[rotations == 2] = np.rot90(map_h[rotations == 2], 2, axes=(1, 2))
    map_h[rotations == -1] = np.rot90(map_h[rotations == -1], 1, axes=(1, 2))
    map_h[rotations == -2] = np.rot90(map_h[rotations == -2], 2, axes=(1, 2))
    return map_h

def get_cropped_maps_heading_exact(world_pts, map, context_size=50, heading=None):
    """world_pts: N x 2 array of positions relative to the world."""
    angles = -heading * 180 / np.pi

    expanded_obs_img = np.full((map.data.shape[0] + context_size, map.data.shape[1] + context_size, map.data.shape[2]),
                               False, dtype=np.float32)
    expanded_obs_img[context_size // 2:-context_size // 2, context_size // 2:-context_size // 2] = map.fdata.astype(
        np.float32)
    img_pts = context_size // 2 + np.round(map.to_map_points(world_pts)).astype(int)
    map_h = np.stack([expanded_obs_img[img_pts[i, 0] - context_size // 2: img_pts[i, 0] + context_size // 2,
                      img_pts[i, 1] - context_size // 2: img_pts[i, 1] + context_size // 2]
                      for i in range(world_pts.shape[0])], axis=0)

    for i in range(angles.shape[0]):
        map_h[i] = interpolation.rotate(map_h[i], reshape=False, angle=angles[i], prefilter=False)

    return map_h