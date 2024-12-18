import random
import SumTree
import numpy as np

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment_per_sampling=0.001):
        self.tree = SumTree.SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.epsilon = 1e-5
        self.size = 0

    def __len__(self):
        return self.size

    def add(self, error, experience):
        priority = (abs(error) + self.epsilon) ** self.alpha
        self.tree.add(priority, experience)
        if self.size < self.capacity:
            self.size += 1

    def sample(self, n):
        batch = []
        idxs = []
        priorities = []
        segment = self.tree.total_priority() / n
        values = np.random.uniform(0, segment, size=n) + np.arange(n) * segment

        for value in values:
            idx, priority, data = self.tree.get_leaf(value)
            batch.append(data)
            idxs.append(idx)
            priorities.append(priority)

        priorities = np.array(priorities)
        sampling_probabilities = priorities / self.tree.total_priority()
        is_weights = np.power(self.capacity * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()

        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)
        return batch, idxs, is_weights

    def update(self, idx, error):
        priority = (abs(error) + self.epsilon) ** self.alpha
        self.tree.update(idx, priority)