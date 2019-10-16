import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from torch.distributions import Categorical

def layer_init(layer, scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer


class FullyConnectedNet(nn.Module):
    def __init__(self, seed, input_size,hidden_size, output_size):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.lin1 = nn.Linear(input_size, hidden_size) # input_size -> hidden_size
        self.relu = nn.ReLU() # RELU
        self.lin2 = nn.Linear(hidden_size, output_size) # FULLY CONNECTED            hidden_sze -> output_size
        self.feature_dim = hidden_size

    def forward(self, x):
        #b = x.view(x.size()[0], -1)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        return F.relu(x)
        

class ProxyPolicyNet(nn.Module):
    def __init__(self, seed, action_size, FC_network):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.FC_network = FC_network
        self.softmax = nn.Softmax(dim=1) # TRY this robsko 
        self.actor = nn.Linear(FC_network.feature_dim, action_size)
        self.critic = nn.Linear(FC_network.feature_dim, 1)


    def forward(self, x):
        x = self.FC_network(x)
        optPolicy = self.actor(x)
        traject = self.softmax(optPolicy)
        value = self.critic(x)
        return traject, value

    def act(self, state, action=None):
        traject, value = self.forward(state)
        max_length = Categorical(traject)
        if action is None:
            action = max_length.sample()
        log_prob = max_length.log_prob(action)
        entropy = max_length.entropy()
        return {'a': action,
                'log_pi_a': log_prob,
                'ent': entropy,
                'v': value.squeeze()}


