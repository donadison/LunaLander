import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import pickle
import os

import DQNNetwork
import PrioritizedReplayBuffer


class DQNAgent:
    def __init__(self, state_size, action_size, device,
                 epsilon_start=0.5,  # Starting exploration probability
                 epsilon_end=0.05,  # Minimum exploration probability
                 epsilon_decay_rate=0.999,  # Epsilon decay rate for exploration-exploitation tradeoff
                 episode_num=None,  # Number of episodes, if provided for epsilon decay
                 memory_capacity=10000,  # Replay memory capacity, default 10000 steps
                 target_update_interval=10  # Interval for updating the target network
                 ):
        self.state_size = state_size  # Size of the state vector
        self.action_size = action_size  # Number of possible actions
        self.device = device  # Device (CPU/GPU)
        self.episode_num = episode_num  # Total number of episodes, for epsilon decay

        # Epsilon exploration parameters
        self.epsilon_start = epsilon_start  # initial epsilon
        self.epsilon = epsilon_start  # current epsilon
        self.epsilon_end = epsilon_end  # minimum epsilon
        self.epsilon_decay_rate = epsilon_decay_rate  # Decay rate for epsilon

        # Primary (online) network and target network
        self.model = DQNNetwork.DQNNetwork(state_size, action_size).to(device)  # online network
        self.target_model = DQNNetwork.DQNNetwork(state_size, action_size).to(device)  # Target network
        self.target_model.load_state_dict(self.model.state_dict())  # Initialize target with primary weights
        self.target_model.eval()  # Target network is not trained

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)  # Adam optimizer for DQN

        # Replay memory with prioritized replay
        self.memory = PrioritizedReplayBuffer.PrioritizedReplayBuffer(capacity=memory_capacity)

        # Other hyperparameters
        self.gamma = 0.99  # Discount rate
        self.target_update_interval = target_update_interval  # How often to update the target network
        self.train_step = 0  # Tracks training steps for updating the target network

    def remember(self, state, action, reward, next_state, done):
        # Compute initial priority (use reward as an approximation for the TD error initially)
        td_error = abs(reward)  # Initial error estimate
        self.memory.add(td_error, (state, action, reward, next_state, done))

    def act(self, state, episode_num=None):
        # Epsilon-greedy action selection
        if episode_num is not None:
            # Linearly decrease epsilon
            self.epsilon = max(
                self.epsilon_end,  # Ensure epsilon does not go below epsilon_end
                self.epsilon_start - (episode_num / self.episode_num) * (self.epsilon_start - self.epsilon_end)
            )

        state = torch.FloatTensor(state).to(self.device)  # Convert state to tensor and move to device
        state = state.unsqueeze(0)  # Add batch dimension

        if episode_num is None:
            # During testing, always select the action with the highest Q-value
            with torch.no_grad():
                act_values = self.model(state)
            return act_values.cpu().argmax().item()

        if np.random.rand() < self.epsilon:
            # Exploration: randomly select an action
            return np.random.choice(self.action_size)
        else:
            # Exploitation: select the action with the highest predicted Q-value
            with torch.no_grad():
                act_values = self.model(state)
            return act_values.cpu().argmax().item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return  # Not enough experiences to train

        # Sample experiences from the replay buffer using prioritized sampling
        batch, idxs, is_weights = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        is_weights = torch.FloatTensor(is_weights).to(self.device)

        q_values = self.model(states)
        current_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN: Use primary network for action selection and target network for Q-value evaluation
        with torch.no_grad():
            next_actions = self.model(next_states).argmax(1)  # Use primary network to select actions
            next_q_values = self.target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(
                1)  # Use target network to evaluate Q-values

            expected_q_values = rewards + (1 - dones) * self.gamma * next_q_values  # Compute the target Q-values

        # Compute TD errors and update priorities
        td_errors = (expected_q_values - current_q_values).detach().cpu().numpy()
        for idx, td_error in zip(idxs, td_errors):
            self.memory.update(idx, td_error)

        # Huber loss for robustness, weighted by importance sampling weights
        loss = (is_weights * F.smooth_l1_loss(current_q_values, expected_q_values, reduction='none')).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network periodically
        self.train_step += 1
        if self.train_step % self.target_update_interval == 0:
            self.target_model.load_state_dict(self.model.state_dict())  # Hard update

    def save(self, filename):
        torch.save(self.model.state_dict(), filename)

    def save_memory(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.memory, f)  # Save the replay buffer
        print("Replay memory saved.")

    def load(self, filename):
        self.model.load_state_dict(torch.load(filename, weights_only=False))
        self.target_model.load_state_dict(self.model.state_dict())  # Ensure target network is synchronized

    def load_memory(self, filename):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.memory = pickle.load(f)  # Load the replay buffer
            print("Replay memory loaded.")
        else:
            print("Replay memory file not found.")
