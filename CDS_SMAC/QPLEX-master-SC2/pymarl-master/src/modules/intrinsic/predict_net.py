import torch
import torch.optim as optim

from torch import nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical, Distribution, Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class LayerNorm(nn.Module):
    """
    Simple 1D LayerNorm.
    """

    def __init__(self, features, center=True, scale=False, eps=1e-6):
        super().__init__()
        self.center = center
        self.scale = scale
        self.eps = eps
        if self.scale:
            self.scale_param = nn.Parameter(torch.ones(features))
        else:
            self.scale_param = None
        if self.center:
            self.center_param = nn.Parameter(torch.zeros(features))
        else:
            self.center_param = None

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        output = (x - mean) / (std + self.eps)
        if self.scale:
            output = output * self.scale_param
        if self.center:
            output = output + self.center_param
        return output


class IVF(nn.Module):

    def __init__(self, num_inputs, hidden_dim, layer_num=3, layer_norm=False):
        super(IVF, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        if layer_num == 3:
            self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.last_fc = nn.Linear(hidden_dim, 1)

        self.layer_norm = layer_norm
        self.layer_num = layer_num
        if layer_norm:
            self.ln1 = LayerNorm(hidden_dim)

        self.apply(weights_init_)

    def forward(self, input):
        if self.layer_norm:
            h = F.relu(self.ln1(self.linear1(input)))
        else:
            h = F.relu(self.linear1(input))

        if self.layer_num == 3:
            h = F.relu(self.linear2(h))
        x = self.last_fc(h)
        return x


class Predict_ID(nn.Module):

    def __init__(self, num_inputs, hidden_dim, n_agents, add_loss_item, lr=1e-3):
        super(Predict_ID, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.last_fc = nn.Linear(hidden_dim, n_agents)

        self.apply(weights_init_)
        self.lr = lr
        self.add_loss_item = add_loss_item
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

        self.CE = nn.CrossEntropyLoss()
        self.CEP = nn.CrossEntropyLoss(reduction='none')

    def forward(self, input):
        h = F.relu(self.linear1(input))
        h = F.relu(self.linear2(h))
        x = torch.softmax(self.last_fc(h), dim=-1)
        return x

    def get_q_id_o(self, obs, id):
        with torch.no_grad():
            predict_ = self.forward(obs)
            log_prob = -1. * self.CEP(predict_, id * torch.ones([obs.shape[0]]).type_as(predict_).long())
            return log_prob.detach()

    def update(self, obs, id):
        predict_ = self.forward(obs)
        loss = self.CE(predict_, id * torch.ones([obs.shape[0]]).type_as(predict_).long())
        obs_c = obs.clone()
        obs_c[1:] = obs[:-1]

        loss += self.add_loss_item * F.mse_loss(predict_, self.forward(obs_c).detach())

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.optimizer.step()


class Predict_Network1(nn.Module):

    def __init__(self, num_inputs, hidden_dim, num_outputs, layer_norm=True, lr=1e-3):
        super(Predict_Network1, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.last_fc = nn.Linear(hidden_dim, num_outputs)

        self.layer_norm = layer_norm
        if layer_norm:
            self.ln1 = LayerNorm(hidden_dim)

        self.apply(weights_init_)
        self.lr = lr

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, input):
        if self.layer_norm:
            h = F.relu(self.ln1(self.linear1(input)))
        else:
            h = F.relu(self.linear1(input))

        h = F.relu(self.linear2(h))
        x = self.last_fc(h)
        return x

    def get_log_pi(self, own_variable, other_variable):
        predict_variable = self.forward(own_variable)
        log_prob = -1 * F.mse_loss(predict_variable, other_variable, reduction='none')
        log_prob = torch.sum(log_prob, -1, keepdim=True)
        return log_prob

    def update(self, own_variable, other_variable, mask):
        predict_variable = self.forward(own_variable)
        loss = F.mse_loss(predict_variable, other_variable, reduction='none')
        loss = loss.sum(dim=-1, keepdim=True)
        loss = (loss * mask).sum() / mask.sum()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.optimizer.step()


class Predict_Network1_combine(nn.Module):

    def __init__(self, num_inputs, hidden_dim, num_outputs, n_agents, layer_norm=True, lr=1e-3):
        super(Predict_Network1_combine, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim + n_agents, hidden_dim)
        self.last_fc = nn.Linear(hidden_dim, num_outputs)

        self.layer_norm = layer_norm
        if layer_norm:
            self.ln1 = LayerNorm(hidden_dim)

        self.apply(weights_init_)
        self.lr = lr
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, input, add_id):
        if self.layer_norm:
            h = F.relu(self.ln1(self.linear1(input)))
        else:
            h = F.relu(self.linear1(input))

        h = torch.cat([h, add_id], dim=-1)
        h = F.relu(self.linear2(h))
        x = self.last_fc(h)
        return x

    def get_log_pi(self, own_variable, other_variable, add_id):
        predict_variable = self.forward(own_variable, add_id)
        log_prob = -1 * F.mse_loss(predict_variable, other_variable, reduction='none')
        log_prob = torch.sum(log_prob, -1, keepdim=True)
        return log_prob

    def update(self, own_variable, other_variable, add_id, mask):
        predict_variable = self.forward(own_variable, add_id)
        loss = F.mse_loss(predict_variable, other_variable, reduction='none')
        loss = loss.sum(dim=-1, keepdim=True)
        loss = (loss * mask).sum() / mask.sum()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.optimizer.step()


class Predict_Network2(nn.Module):

    def __init__(self, num_inputs, hidden_dim, num_components=4, layer_norm=True, lr=1e-3):
        super(Predict_Network2, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.layer_norm = layer_norm
        if layer_norm:
            self.ln1 = LayerNorm(hidden_dim)

        self.mean_list = []
        for _ in range(num_components):
            self.mean_list.append(nn.Linear(hidden_dim, num_inputs))

        self.mean_list = nn.ModuleList(self.mean_list)
        self.num_components = num_components
        self.com_last_fc = nn.Linear(hidden_dim, num_components)

        self.apply(weights_init_)
        self.lr = lr

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, input):

        if self.layer_norm:
            x1 = F.relu(self.ln1(self.linear1(input)))
        else:
            x1 = F.relu(self.linear1(input))

        x2 = F.relu(self.linear2(x1))
        com_h = torch.softmax(self.com_last_fc(x2), dim=-1)

        means, stds = [], []
        for i in range(self.num_components):
            mean = self.mean_list[i](x2)
            means.append(mean)
            stds.append(torch.ones_like(mean))

        return com_h, means, stds

    def get_log_pi(self, own_variable, other_variable):
        com_h, means, stds = self.forward(own_variable)
        mix = Categorical(logits=com_h)
        means = torch.stack(means, 1)
        stds = torch.stack(stds, 1)

        comp = torch.distributions.independent.Independent(Normal(means, stds), 1)
        gmm = torch.distributions.mixture_same_family.MixtureSameFamily(mix, comp)

        return gmm.log_prob(other_variable)


class Predict_Network3(nn.Module):

    def __init__(self, num_inputs, hidden_dim, num_components=4, layer_norm=True, lr=1e-3):
        super(Predict_Network3, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.layer_norm = layer_norm
        if layer_norm:
            self.ln1 = LayerNorm(hidden_dim)

        self.mean_list = []
        for _ in range(num_components):
            self.mean_list.append(nn.Linear(hidden_dim, num_inputs))

        self.log_std_list = []
        for _ in range(num_components):
            self.log_std_list.append(nn.Linear(hidden_dim, num_inputs))

        self.mean_list = nn.ModuleList(self.mean_list)
        self.log_std_list = nn.ModuleList(self.log_std_list)
        self.num_components = num_components
        self.com_last_fc = nn.Linear(hidden_dim, num_components)

        self.apply(weights_init_)
        self.lr = lr

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, input):

        if self.layer_norm:
            x1 = F.relu(self.ln1(self.linear1(input)))
        else:
            x1 = F.relu(self.linear1(input))

        x2 = F.relu(self.linear2(x1))
        com_h = torch.softmax(self.com_last_fc(x2), dim=-1)

        means, stds = [], []
        for i in range(self.num_components):
            mean = self.mean_list[i](x2)
            log_std = self.log_std_list[i](x2)

            means.append(mean)
            stds.append(log_std.exp())

        return com_h, means, stds

    def get_log_pi(self, own_variable, other_variable):
        com_h, means, stds = self.forward(own_variable)
        mix = Categorical(logits=com_h)
        means = torch.stack(means, 1)
        stds = torch.stack(stds, 1)

        comp = torch.distributions.independent.Independent(Normal(means, stds), 1)
        gmm = torch.distributions.mixture_same_family.MixtureSameFamily(mix, comp)

        return gmm.log_prob(other_variable)


def get_predict_model(num_inputs, hidden_dim, model_id, layer_norm=True):
    if model_id == 1:
        return Predict_Network1(num_inputs, hidden_dim, layer_norm=layer_norm)
    elif model_id == 2:
        return Predict_Network2(num_inputs, hidden_dim, layer_norm=layer_norm)
    elif model_id == 3:
        return Predict_Network3(num_inputs, hidden_dim, layer_norm=layer_norm)
    else:
        raise (print('error predict model'))
