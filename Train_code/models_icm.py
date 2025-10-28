import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(23, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 2)

    def forward(self, s):
        h_fc1 = F.relu(self.fc1(s))
        h_fc2 = F.relu(self.fc2(h_fc1))
        h_fc3 = F.relu(self.fc3(h_fc2))
        mu = torch.tanh(self.fc4(h_fc3))
        return mu

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6
class PolicyNetGaussian(nn.Module):
    def __init__(self):
        super(PolicyNetGaussian, self).__init__()
        self.fc1 = nn.Linear(23, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4_mean = nn.Linear(512, 2)
        self.fc4_logstd = nn.Linear(512, 2)

    def forward(self, s):
        h_fc1 = F.relu(self.fc1(s))
        h_fc2 = F.relu(self.fc2(h_fc1))
        h_fc3 = F.relu(self.fc3(h_fc2))
        a_mean = self.fc4_mean(h_fc3)
        a_logstd = self.fc4_logstd(h_fc3)
        a_logstd = torch.clamp(a_logstd, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return a_mean, a_logstd
    
    def sample(self, s):
        a_mean, a_logstd = self.forward(s)
        a_std = a_logstd.exp()
        normal = Normal(a_mean, a_std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)

        # Enforcing action Bound
        log_prob -= torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, torch.tanh(a_mean)

class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(23, 512)
        self.fc2 = nn.Linear(512+2, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 1)
    
    def forward(self, s, a):
        h_fc1 = F.relu(self.fc1(s))
        h_fc1_a = torch.cat((h_fc1, a), 1)
        h_fc2 = F.relu(self.fc2(h_fc1_a))
        h_fc3 = F.relu(self.fc3(h_fc2))
        q_out = self.fc4(h_fc3)
        return q_out
##########################################################
class ICM(nn.Module):
    # Add swish activation
    def __init__(self):
        super(ICM, self).__init__()

        # Inverse model
        self.fc1 = nn.Linear(23, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256+256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 2)



        # Forward model
        self.fc6 = nn.Linear(256+2, 256)
        # self.fc7 = nn.Linear(23, 256)
        # self.fc8 = nn.Linear(256 + 2, 256)


    def forward(self, a, s, sn):
        # Inverse model
        h_fc1 = F.relu(self.fc1(s))
        h_fc1 = F.relu(self.fc2(h_fc1))
        # print('------++++++h_fc1',h_fc1)
        h_fc2 = F.relu(self.fc1(sn))
        h_fc2 = F.relu(self.fc2(h_fc2))
        inverse_vec = torch.cat((h_fc1, h_fc2), 1)
        # print('------++++++h_fc_out', h_fc_out)
        h_fc3 = F.relu(self.fc3(inverse_vec))
        h_fc4 = F.relu(self.fc4(h_fc3))
        pred_a = F.relu(self.fc5(h_fc4))


        ##############################
        # loss_fn = nn.MSELoss(reduction='mean')
        # input = torch.autograd.Variable(torch.from_numpy(pred_a))
        # target = torch.autograd.Variable(torch.from_numpy(a))
        # print('99999999999target',target)
        # inv_loss = loss_fn(input.float(), target.float())
        inv_loss_a = nn.MSELoss(reduction='none')(pred_a, a.detach())
        # print('-------------------inv_loss_a', a, inv_loss_a)
        inv_loss = inv_loss_a.mean()
        # print('+++++++++++++++++++inv_loss',inv_loss)
        # inv_loss1 = nn.MSELoss()(pred_a, a)
        #
        # # inv_loss1 = (nn.MSELoss(pred_a, a)).mean()
        # print('9999999999999999999999999999---------inv_loss', inv_loss1)


        # Forward model
        # h_fc6 = F.relu(self.fc6(sn))
        # # print('------++++++h_fc6',h_fc6,len(h_fc6))
        # h_fc7 = F.relu(self.fc7(s))
        # h_fc7_a = torch.cat((h_fc7, a), 1)
        # h_fc8 = F.relu(self.fc8(h_fc7_a))

        forward_vec = torch.cat((h_fc1, a), 1).detach()
        h_fc6 = F.relu(self.fc6(forward_vec))


        # Intrinsic reward
        intr_reward = 0.5 * nn.MSELoss(reduction='none')(h_fc2.detach(), h_fc6)
        # intr_reward1 = 0.5 * nn.MSELoss()(h_fc8, h_fc6)
        # print('000000000000intr_reward0',intr_reward, intr_reward.size(), len(intr_reward))
        intr_rewards = intr_reward.mean(-1)
        # print('11111111111111111intr_rewards',intr_rewards, len(intr_rewards))
        intr_rewards = intr_rewards.unsqueeze(-1)
        # print('//////////////intr_rewards',intr_rewards,len(intr_rewards),intr_rewards.shape)

        # Forward loss
        forw_loss = intr_rewards.mean()

        # print('88888888888forw_loss',forw_loss)

        return intr_rewards, inv_loss, forw_loss

#############################################################################
# class ICM(nn.Module):
#     # Add swish activation
#     def __init__(self):
#         super(ICM, self).__init__()
#
#         # Inverse model
#         self.fc1 = nn.Linear(23, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256+256, 256)
#         self.fc4 = nn.Linear(256, 256)
#         self.fc5 = nn.Linear(256, 2)
#
#
#
#         # Forward model
#         self.fc6 = nn.Linear(256+2, 256)
#         # self.fc7 = nn.Linear(23, 256)
#         # self.fc8 = nn.Linear(256 + 2, 256)
#
#
#     def forward(self, a, s, sn):
#         # Inverse model
#         h_fc1 = F.relu(self.fc1(s))
#         h_fc1 = F.relu(self.fc2(h_fc1))
#         # print('------++++++h_fc1',h_fc1)
#         h_fc2 = F.relu(self.fc1(sn))
#         h_fc2 = F.relu(self.fc2(h_fc2))
#
#         h_fc_out = torch.cat((h_fc1, h_fc2), 1)
#         # print('------++++++h_fc_out', h_fc_out)
#         h_fc3 = F.relu(self.fc3(h_fc_out))
#         h_fc4 = F.relu(self.fc4(h_fc3))
#         pred_a = F.relu(self.fc5(h_fc4))
#         # print('--------------pre_a',pred_a)
#         # print('+++++++++++++++++a',a)
#
#
#         ##############################
#         # loss_fn = nn.MSELoss(reduction='mean')
#         # input = torch.autograd.Variable(torch.from_numpy(pred_a))
#         # target = torch.autograd.Variable(torch.from_numpy(a))
#         # print('99999999999target',target)
#         # inv_loss = loss_fn(input.float(), target.float())
#
#         # inv_loss_a = nn.MSELoss(reduction='none')(pred_a, a)
#         # # print('-------------------inv_loss_a', inv_loss_a)
#         # inv_loss = inv_loss_a.mean()
#         # print('+++++++++++++++++++inv_loss',inv_loss)
#
#         # inv_loss = (nn.MSELoss(pred_a, a)).mean()
#         # print('9999999999999999999999999999---------inv_loss', inv_loss)
#
#
#         # Forward model
#
#         h_fc6_a = torch.cat((h_fc1, a), 1)
#         h_fc6 = F.relu(self.fc6(h_fc6_a))
#
#
#         # Intrinsic reward
#         intr_reward = 0.5 * nn.MSELoss(reduction='none')(h_fc6, h_fc2)
#         # intr_reward1 = 0.5 * nn.MSELoss()(h_fc8, h_fc6)
#         # print('11111111111111111intr_reward1',intr_reward1)
#         intr_rewards = intr_reward.mean(-1)
#         intr_rewards = intr_rewards.unsqueeze(-1)
#         # print('//////////////intr_rewards',intr_rewards,len(intr_rewards),intr_rewards.shape)
#
#         # Forward loss
#         forw_loss = intr_rewards.mean()
#
#         # print('88888888888forw_loss',forw_loss)
#
#         return intr_rewards, forw_loss
