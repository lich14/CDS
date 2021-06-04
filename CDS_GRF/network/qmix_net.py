import torch.nn as nn
import torch
import torch.nn.functional as F


class QMixNet(nn.Module):

    def __init__(self, args):
        super(QMixNet, self).__init__()
        self.args = args
        if args.two_hyper_layers:
            self.hyper_w1 = nn.Sequential(
                nn.Linear(args.state_shape, args.hyper_hidden_dim), nn.ReLU(),
                nn.Linear(args.hyper_hidden_dim, args.n_agents * args.qmix_hidden_dim))
            self.hyper_w2 = nn.Sequential(
                nn.Linear(args.state_shape, args.hyper_hidden_dim), nn.ReLU(), nn.Linear(args.hyper_hidden_dim, args.qmix_hidden_dim))
        else:
            self.hyper_w1 = nn.Linear(args.state_shape, args.n_agents * args.qmix_hidden_dim)
            self.hyper_w2 = nn.Linear(args.state_shape, args.qmix_hidden_dim * 1)

        self.hyper_b1 = nn.Linear(args.state_shape, args.qmix_hidden_dim)
        self.hyper_b2 = nn.Sequential(nn.Linear(args.state_shape, args.qmix_hidden_dim), nn.ReLU(), nn.Linear(args.qmix_hidden_dim, 1))

    def forward(self, q_values, states):
        episode_num = q_values.size(0)
        q_values = q_values.view(-1, 1, self.args.n_agents)
        states = states.reshape(-1, self.args.state_shape)  # (1920, 120)

        w1 = torch.abs(self.hyper_w1(states))  # (1920, 160)
        b1 = self.hyper_b1(states)  # (1920, 32)

        w1 = w1.view(-1, self.args.n_agents, self.args.qmix_hidden_dim)  # (1920, 5, 32)
        b1 = b1.view(-1, 1, self.args.qmix_hidden_dim)  # (1920, 1, 32)

        hidden = F.elu(torch.bmm(q_values, w1) + b1)  # (1920, 1, 32)

        w2 = torch.abs(self.hyper_w2(states))  # (1920, 32)
        b2 = self.hyper_b2(states)  # (1920, 1)

        w2 = w2.view(-1, self.args.qmix_hidden_dim, 1)  # (1920, 32, 1)
        b2 = b2.view(-1, 1, 1)  # (1920, 1， 1)

        q_total = torch.bmm(hidden, w2) + b2  # (1920, 1, 1)
        q_total = q_total.view(episode_num, -1, 1)  # (32, 60, 1)

        return q_total
