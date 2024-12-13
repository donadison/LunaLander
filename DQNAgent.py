import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import pickle
import os

import DQNNetwork


class DQNAgent:
    def __init__(self, state_size, action_size, device,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay_rate=0.999,
                 max_steps=None):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.max_steps = max_steps

        # Epsilon exploration parameters
        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_rate = epsilon_decay_rate

        # Neural network and optimizer
        self.model = DQNNetwork.DQNNetwork(state_size, action_size).to(device)
        self.target_model = DQNNetwork.DQNNetwork(state_size, action_size).to(device)
        self.target_model.load_state_dict(self.model.state_dict())  # Initialize target network
        self.target_model.eval()  # Target model is in evaluation mode

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Replay memory
        self.memory = deque(maxlen=50000)

        # Other hyperparameters
        self.gamma = 0.95  # Discount rate
        self.tau = 0.01  # Soft update factor for target network

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, episode_num=None):
        if episode_num is not None:
            # Define the decay factor and adjust the range dynamically
            decay_rate = -np.log(self.epsilon_end / self.epsilon_start) / self.max_steps
            self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(
                -decay_rate * episode_num)

        state = torch.FloatTensor(state).to(self.device)
        if np.random.rand() < self.epsilon:
            # Explore: randomly choose an action
            return np.random.choice(self.action_size)
        else:
            # Exploit: choose the best action from the model
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

        # Compute Q-values for current states
        q_values = self.model(states)
        current_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute next Q-values using the target network
        next_actions = self.model(next_states).argmax(1)  # Actions from policy network
        next_q_values = self.target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)

        # Compute target Q-values
        expected_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Huber loss
        loss = F.smooth_l1_loss(current_q_values, expected_q_values.detach(), beta=1.0)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Perform soft updates on target network
        self.soft_update_target_network()

    def soft_update_target_network(self):
        """Soft update the target network parameters."""
        for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.model.state_dict(), filename)

    def save_memory(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.memory, f)
        print("Replay memory saved.")

    def load(self, filename):
        self.model.load_state_dict(torch.load(filename))
        self.target_model.load_state_dict(torch.load(filename))  # Sync target model

    def load_memory(self, filename):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.memory = pickle.load(f)
            print("Replay memory loaded.")
        else:
            print("Replay memory file not found.")