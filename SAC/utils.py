import math
import random

import numpy as np
import torch
import torch.nn as nn


class ExperienceReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.reward = np.zeros((max_size, 1))
        self.next_state = np.zeros((max_size, state_dim))
        self.done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, reward, next_state, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        index = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[index]).to(self.device),
            torch.FloatTensor(self.action[index]).to(self.device),
            torch.FloatTensor(self.reward[index]).to(self.device),
            torch.FloatTensor(self.next_state[index]).to(self.device),
            torch.FloatTensor(self.done[index]).to(self.device)
        )

    def remove(self, up_to_idx):
        self.state = self.state[up_to_idx:]
        self.action = self.action[up_to_idx:]
        self.reward = self.reward[up_to_idx:]
        self.next_state = self.next_state[up_to_idx:]
        self.done = self.done[up_to_idx:]

        self.ptr -= up_to_idx
        self.size -= up_to_idx

    def __len__(self):
        return self.size


class ParameterExperienceReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.action_prob = np.zeros((max_size, 1))
        self.reward = np.zeros((max_size, 1))
        self.next_state = np.zeros((max_size, state_dim))
        self.done = np.zeros((max_size, 1))
        self.mean = np.zeros((max_size, action_dim))
        self.std = np.zeros((max_size, action_dim))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, action_prob, reward, next_state, done, mean, std):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.action_prob[self.ptr] = action_prob
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done
        self.mean[self.ptr] = mean
        self.std[self.ptr] = std

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        index = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[index]).to(self.device),
            torch.FloatTensor(self.action[index]).to(self.device),
            torch.FloatTensor(self.action_prob[index]).to(self.device),
            torch.FloatTensor(self.reward[index]).to(self.device),
            torch.FloatTensor(self.next_state[index]).to(self.device),
            torch.FloatTensor(self.done[index]).to(self.device),
            torch.FloatTensor(self.mean[index]).to(self.device),
            torch.FloatTensor(self.std[index]).to(self.device)
        )

    def remove(self, up_to_idx):
        self.state = self.state[up_to_idx:]
        self.action = self.action[up_to_idx:]
        self.action_prob = self.action_prob[up_to_idx:]
        self.reward = self.reward[up_to_idx:]
        self.next_state = self.next_state[up_to_idx:]
        self.done = self.done[up_to_idx:]
        self.mean = self.mean[up_to_idx:]
        self.std = self.std[up_to_idx:]

        self.ptr -= up_to_idx
        self.size -= up_to_idx

    def __len__(self):
        return self.size


def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p


def log_sum_exp(inputs, dim=None, keep_dim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keep_dim:
        outputs = outputs.squeeze(dim)
    return outputs


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)
