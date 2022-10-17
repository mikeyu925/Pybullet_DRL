import torch
import torch.nn as nn
import torch.nn.functional as F

# Actor网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 200)
        self.l2 = nn.Linear(200, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action
    # return action
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x