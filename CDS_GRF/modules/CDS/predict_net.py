from numpy import add
import torch
import torch.optim as optim

from torch import nn as nn
from torch.nn import functional as F

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class Predict_Network(nn.Module):

    def __init__(self, num_inputs, hidden_dim, num_outputs, lr=3e-4):
        super(Predict_Network, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.last_fc = nn.Linear(hidden_dim, num_outputs)

        self.apply(weights_init_)
        self.lr = lr

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, input):
        h = F.relu(self.linear1(input))
        h = F.relu(self.linear2(h))
        x = self.last_fc(h)
        return x

    def get_log_pi(self, own_variable, other_variable):
        predict_variable = self.forward(own_variable)
        log_prob = -1 * F.mse_loss(predict_variable,
                                   other_variable, reduction='none')
        log_prob = torch.sum(log_prob, -1, keepdim=True)
        return log_prob

    def update(self, own_variable, other_variable, mask):
        if mask.sum() > 0:
            predict_variable = self.forward(own_variable)
            loss = F.mse_loss(predict_variable,
                              other_variable, reduction='none')
            loss = loss.sum(dim=-1, keepdim=True)
            loss = (loss * mask).sum() / mask.sum()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.)
            self.optimizer.step()

            return loss.to('cpu').detach().item()

        return None


class Predict_Network_WithID(nn.Module):

    def __init__(self, num_inputs, hidden_dim, num_outputs, n_agents, lr=3e-4):
        super(Predict_Network_WithID, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim + n_agents, hidden_dim)
        self.last_fc = nn.Linear(hidden_dim, num_outputs)

        self.apply(weights_init_)
        self.lr = lr
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, inputs, add_id):
        inputs = torch.cat([inputs, add_id], dim=-1)
        h = F.relu(self.linear1(inputs))

        h = torch.cat([h, add_id], dim=-1)
        h = F.relu(self.linear2(h))
        x = self.last_fc(h)
        return x

    def get_log_pi(self, own_variable, other_variable, add_id):
        predict_variable = self.forward(own_variable, add_id)
        log_prob = -1 * F.mse_loss(predict_variable,
                                   other_variable, reduction='none')
        log_prob = torch.sum(log_prob, -1, keepdim=True)
        return log_prob

    def update(self, own_variable, other_variable, add_id, mask):
        if mask.sum() > 0:
            predict_variable = self.forward(own_variable, add_id)
            loss = F.mse_loss(predict_variable,
                              other_variable, reduction='none')
            loss = loss.sum(dim=-1, keepdim=True)
            loss = (loss * mask).sum() / mask.sum()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.)
            self.optimizer.step()

            return loss.to('cpu').detach().item()

        return None


class Predict_ID_obs_tau(nn.Module):

    def __init__(self, tau_dim, hidden_dim, n_agents, lr=1e-3):
        super(Predict_ID_obs_tau, self).__init__()

        self.linear1 = nn.Linear(tau_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.last_fc = nn.Linear(hidden_dim, n_agents)

        self.apply(weights_init_)
        self.lr = lr
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

        self.CE = nn.CrossEntropyLoss()
        self.CEP = nn.CrossEntropyLoss(reduction='none')

    def forward(self, tau):
        h = F.relu(self.linear1(tau))
        h = F.relu(self.linear2(h))
        x = torch.softmax(self.last_fc(h), dim=-1)
        return x

    def update(self, tau, id, mask):

        if mask.sum() > 0:

            predict_ = self.forward(tau)
            loss = self.CEP(predict_, id.long())
            loss = (loss * mask).sum() / mask.sum()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.)
            self.optimizer.step()

            return loss.to('cpu').detach().item()

        return None
