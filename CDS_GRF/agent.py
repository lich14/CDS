import numpy as np
import torch
from policy.qplex_sdq_intrinsic import DMAQ_qattenLearner_SDQ_intrinsic
from torch.distributions import Categorical


class Agents:

    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.policy = DMAQ_qattenLearner_SDQ_intrinsic(args)

        self.args = args
        print('Init Agents')

    def _get_max_episode_len(self, batch):
        terminated = batch['terminated']
        episode_num = terminated.shape[0]
        max_episode_len = 0
        for episode_idx in range(episode_num):
            for transition_idx in range(self.args.episode_limit):
                if terminated[episode_idx, transition_idx, 0] == 1:
                    if transition_idx + 1 >= max_episode_len:
                        max_episode_len = transition_idx + 1
                    break
        return max_episode_len

    def train(self, batch, train_step, t_env, epsilon=None):
        max_episode_len = self._get_max_episode_len(batch)
        for key in batch.keys():
            batch[key] = batch[key][:, :max_episode_len]
        loss = self.policy.learn(batch, max_episode_len, train_step, t_env, epsilon)
        if train_step > 0 and train_step % self.args.save_cycle == 0:
            self.policy.save_model(train_step)

        return loss
