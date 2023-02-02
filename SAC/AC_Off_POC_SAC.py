import os

from utils import soft_update, hard_update, weights_init_

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.kl import kl_divergence

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class Actor(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(Actor, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class Critic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)

        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, device, action_space=None):
        super(GaussianPolicy, self).__init__()
        self.device = device

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.).to(self.device)
            self.action_bias = torch.tensor(0.).to(self.device)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.).to(self.device)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.).to(self.device)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        initial_mean = mean
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, mean, initial_mean, std

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class AC_Off_POC_SAC(object):
    def __init__(self, num_inputs, action_space, args, device):
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = device

        self.critic = Critic(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = Critic(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
        if self.automatic_entropy_tuning is True:
            self.target_entropy = -torch.prod(torch.Tensor(action_space.shape)).to(self.device).item()
            self.log_alpha = torch.zeros(1, requires_grad=True).to(self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

        self.actor = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, self.device, action_space).to(self.device)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=args.lr)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        if evaluate:
            _, _, mean, _, _ = self.actor.sample(state)

            mean = mean.detach().cpu().numpy()[0]

            return mean
        else:
            action, log_prob, _, mean, std = self.actor.sample(state)

            action = action.detach().cpu().numpy()[0]
            action_prob = log_prob.exp().detach().cpu().numpy()[0]
            mean, std = mean.detach().cpu().numpy()[0], std.detach().cpu().numpy()[0]

            return action, action_prob, mean, std

    def update_parameters(self, memory, batch_size, updates, off_poc_update):
        # Sample from the experience replay buffer
        state_batch, action_batch, action_prob_batch, reward_batch, next_state_batch, mask_batch, mean_batch, std_batch = memory.sample(batch_size=batch_size)

        # state_batch = torch.FloatTensor(state_batch).to(self.device)
        # next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        # action_batch = torch.FloatTensor(action_batch).to(self.device)
        # action_prob_batch = torch.FloatTensor(action_prob_batch).to(self.device)
        # reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        # mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        if off_poc_update:
            # Compute JS-Divergence weights
            try:
                # mean_batch = torch.FloatTensor(mean_batch).to(self.device)
                # std_batch = torch.FloatTensor(std_batch).to(self.device)

                with torch.no_grad():
                    _, log_prob, _, mean, std = self.actor.sample(state_batch)

                current_action_prob = log_prob.exp()
                action_prob_ratio = current_action_prob / action_prob_batch

                current_distribution = Normal(mean, std)
                batch_distribution = Normal(mean_batch, std_batch)

                js_div = (kl_divergence(batch_distribution, current_distribution) + kl_divergence(current_distribution, batch_distribution)) / 2
                js_div = torch.mean(torch.exp(-js_div), dim=1).view(-1, 1)

                js_weights = torch.min(action_prob_ratio, js_div)
            except:
                js_weights = torch.ones_like(reward_batch)
        else:
            js_weights = torch.ones_like(reward_batch)

        with torch.no_grad():
            # Select the target smoothing regularized action according to policy
            next_state_action, next_state_log_pi, _, _, _ = self.actor.sample(next_state_batch)

            # Compute the target Q-value
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)

        # Get the current Q-value estimates
        qf1, qf2 = self.critic(state_batch, action_batch)

        # Compute the critic loss
        qf1_loss = torch.sum(js_weights * F.mse_loss(qf1, next_q_value, reduction='none')) / torch.sum(
            js_weights)
        qf2_loss = torch.sum(js_weights * F.mse_loss(qf2, next_q_value, reduction='none')) / torch.sum(
            js_weights)
        qf_loss = qf1_loss + qf2_loss

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        qf_loss.backward()
        self.critic_optimizer.step()

        # Compute policy loss
        pi, log_pi, _, _, _ = self.actor.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = (js_weights * ((self.alpha * log_pi) - min_qf_pi)).sum() / torch.sum(js_weights)

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # Tune the temperature coefficient
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()

        # Soft update the target critic network
        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

    # Save the model parameters
    def save(self, file_name):
        torch.save(self.actor.state_dict(), file_name + "_actor")
        torch.save(self.actor_optimizer.state_dict(), file_name + "_actor_optimizer")

        torch.save(self.critic.state_dict(), file_name + "_critic")
        torch.save(self.critic_optimizer.state_dict(), file_name + "_critic_optimizer")

    # Load the model parameters
    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))

        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = hard_update.deepcopy(self.critic)