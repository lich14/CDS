"""multi-head attention model that calculate local Q-function."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from network.predict_net import weights_init_


class ATT_MLP(nn.Module):

    def __init__(self, num_inputs, hidden_dim, embed_dim):
        super(ATT_MLP, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.last_fc = nn.Linear(hidden_dim, embed_dim)
        self.apply(weights_init_)

    def forward(self, state):
        h = F.relu(self.linear1(state))
        x = self.last_fc(h)
        return x