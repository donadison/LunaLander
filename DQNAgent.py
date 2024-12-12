import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import deque


import DQNNetwork

class DQNAgent:
    def __init__(self, state_size, action_size, device,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay_rate=0.999):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device

        # Epsilon exploration parameters
        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_rate = epsilon_decay_rate
        self.steps = 0

        # Neural network and optimizer
        self.model = DQNNetwork.DQNNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Replay memory
        self.memory = deque(maxlen=10000)

        # Other hyperparameters
        self.gamma = 0.95  # Discount rate

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Increment steps for custom decay
        if self.epsilon == self.epsilon_end:
            self.epsilon = self.epsilon_start

        self.steps += 1

        # Exponential decay
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay_rate)

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            act_values = self.model(state)
        return act_values.cpu().argmax().item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # Convert to tensors and ensure correct shape
        states = torch.FloatTensor(np.array(states)).to(self.device).squeeze(1)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device).squeeze(1)
        dones = torch.FloatTensor(dones).to(self.device)

        # Compute Q-values
        q_values = self.model(states)
        current_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute next Q-values
        next_q_values = self.model(next_states).max(1)[0]

        # Compute target Q-values
        expected_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss
        loss = F.mse_loss(current_q_values, expected_q_values.detach())

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, filename):
        torch.save(self.model.state_dict(), filename)

    def load(self, filename):
        self.model.load_state_dict(torch.load(filename))