import torch.nn as nn
from torch import sigmoid as sig


class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNNetwork, self).__init__()
        self.layer1 = nn.Linear(state_size, 1024)
        self.layer2 = nn.Linear(1024, 1024)
        self.layer3 = nn.Linear(1024, 512)
        self.layer4 = nn.Linear(512, action_size)
        self.swish = nn.SiLU()  # Swish activation function

        # Xavier initialization for weight initialization
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.xavier_uniform_(self.layer3.weight)
        nn.init.xavier_uniform_(self.layer4.weight)
        self.layer1.bias.data.fill_(0.01)
        self.layer2.bias.data.fill_(0.01)
        self.layer3.bias.data.fill_(0.01)
        self.layer4.bias.data.fill_(0.01)

    def forward(self, x):
        # Forward pass through the network
        x = self.swish(self.layer1(x))  # Apply Swish activation after layer1
        x = self.swish(self.layer2(x))  # Apply Swish activation after layer2
        x = self.swish(self.layer3(x))  # Apply Swish activation after layer3
        return self.layer4(x)  # Output layer (Q-values for each action)
