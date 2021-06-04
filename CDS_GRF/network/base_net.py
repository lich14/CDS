import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np


class Uniform:

    def __init__(self, args):
        self.args = args
        self.device = torch.device('cpu')
        self.noise_distrib = torch.distributions.one_hot_categorical.OneHotCategorical(
            torch.tensor([1 / self.args.noise_dim for _ in range(self.args.noise_dim)]))

    def sample(self, state, test_mode):
        return self.noise_distrib.sample().to(self.device)

    def update_returns(self, state, noise, returns, test_mode, t):
        pass

    def to(self, device):
        self.device = device


class RNN(nn.Module):

    def __init__(self, input_shape, args):
        super(RNN, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        # self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRU(
            input_size=args.rnn_hidden_dim,
            num_layers=1,
            hidden_size=args.rnn_hidden_dim,
            batch_first=True,
        )

        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, obs, hidden_state):
        if len(hidden_state.shape) == 2:
            hidden_state = hidden_state.unsqueeze(0)

        obs_c = obs.view(-1, obs.shape[-1])
        x = f.relu(self.fc1(obs_c))
        x = x.reshape(obs.shape[0], obs.shape[1], -1)

        h_in = hidden_state
        gru_out, _ = self.rnn(x, h_in)
        gru_out_c = gru_out.reshape(-1, gru_out.shape[-1])
        q = self.fc2(gru_out_c)
        q = q.reshape(obs.shape[0], obs.shape[1], -1)

        return q, gru_out


class MLP(nn.Module):

    def __init__(self, args):
        super(MLP, self).__init__()
        self.args = args
        self.fc = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def forward(self, hidden_state):
        q = self.fc(hidden_state)
        return q


class MLP_2(nn.Module):

    def __init__(self, args):
        super(MLP_2, self).__init__()
        self.args = args
        self.fc_1 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc_2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def forward(self, hidden_state):
        h1 = f.relu(self.fc_1(hidden_state))
        q = self.fc_2(h1)
        return q


class Critic(nn.Module):

    def __init__(self, input_shape, args):
        super(Critic, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, args.critic_dim)
        self.fc2 = nn.Linear(args.critic_dim, args.critic_dim)
        self.fc3 = nn.Linear(args.critic_dim, 1)

    def forward(self, inputs):
        x = f.relu(self.fc1(inputs))
        x = f.relu(self.fc2(x))
        q = self.fc3(x)
        return q
