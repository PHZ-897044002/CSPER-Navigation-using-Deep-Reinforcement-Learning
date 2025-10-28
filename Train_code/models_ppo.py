import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6
class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(23, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4_mean = nn.Linear(512, 2)
        self.fc4_logstd = nn.Linear(512, 2)

    def forward(self, s):
        h_fc1 = F.relu(self.fc1(s))
        h_fc2 = F.relu(self.fc2(h_fc1))
        h_fc3 = F.relu(self.fc3(h_fc2))
        a_mean = torch.tanh(self.fc4_mean(h_fc3))
        print('*********************a_mean',a_mean)
        # a_mean = self.fc4_mean(h_fc3)
        a_logstd = self.fc4_logstd(h_fc3)
        a_logstd = torch.clamp(a_logstd, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        print('*********************a_logstd', a_logstd)
        return a_mean, a_logstd
    
    def distribution(self, s):
        a_mean, a_logstd = self.forward(s)
        a_std = a_logstd.exp()
        #a_std = 0.8*torch.ones(a_mean.shape, dtype=torch.float64).cuda()
        dist = Normal(a_mean, a_std)
        return dist

    def sample(self, s):
        dist = self.distribution(s)
        # a_samp = dist.sample()
        ###################################
        x_t = dist.rsample()
        action = torch.tanh(x_t)
        # a_samp = torch.clamp(a_samp, min=-1, max=1)   # action clamp -1 and 1
        ####################################
        logp = dist.log_prob(action)

        return action, logp

class ValueNet(nn.Module):
    def __init__(self):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(23, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 1)
    
    def forward(self, s):
        h_fc1 = F.relu(self.fc1(s))
        h_fc2 = F.relu(self.fc2(h_fc1))
        h_fc3 = F.relu(self.fc3(h_fc2))
        v_out = self.fc4(h_fc3)
        return v_out