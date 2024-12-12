import torch.nn as nn
import torch.nn.functional as F


class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.value_fc = nn.Linear(32, 1)  # State-value stream
        self.advantage_fc = nn.Linear(32, action_size)  # Action-advantage stream
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.value_fc.weight)
        nn.init.xavier_uniform_(self.advantage_fc.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.value_fc(x)
        advantage = self.advantage_fc(x)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))
