import random
import SumTree
import numpy as np


class PrioritizedReplayBuffer:
    # A prioritized replay buffer for storing experiences with importance-sampling.
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment_per_sampling=0.001):
        self.tree = SumTree.SumTree(capacity)  # Initialize the SumTree for efficient prioritized sampling
        self.capacity = capacity  # Capacity of the replay buffer
        self.alpha = alpha  # Exponent for prioritization (controls how much error contributes to priorities)
        self.beta = beta  # Initial beta value for importance sampling
        self.beta_increment_per_sampling = beta_increment_per_sampling  # Increment value for beta
        self.epsilon = 1e-5  # Small value added to avoid zero priorities
        self.size = 0  # Current number of experiences in the buffer

    def __len__(self):
        return self.size  # overload for the return the current size of the buffer

    def add(self, error, experience):
        # Add a new experience with its TD error to the buffer
        priority = (abs(error) + self.epsilon) ** self.alpha  # Compute priority based on TD error
        self.tree.add(priority, experience)  # Add to the SumTree with the calculated priority
        if self.size < self.capacity:  # Ensure the buffer size does not exceed capacity
            self.size += 1

    def sample(self, n):
        # Sample `n` experiences from the buffer based on priorities
        batch = []  # List to store sampled experiences
        idxs = []  # Indices of sampled experiences
        priorities = []  # List to store priorities of sampled experiences
        segment = self.tree.total_priority() / n  # Divide the total priority into `n` segments
        values = np.random.uniform(0, segment, size=n) + np.arange(n) * segment  # Random positions within segments

        for value in values:
            idx, priority, data = self.tree.get_leaf(
                value)  # Get the leaf node from SumTree corresponding to the segment
            batch.append(data)  # Append the experience to the batch
            idxs.append(idx)  # Append the index of the experience
            priorities.append(priority)  # Append the priority of the experience

        priorities = np.array(priorities)  # Convert priorities to NumPy array
        sampling_probabilities = priorities / self.tree.total_priority()  # Normalize priorities for sampling
        is_weights = np.power(self.capacity * sampling_probabilities, -self.beta)  # Importance-sampling weights
        is_weights /= is_weights.max()  # Normalize weights to prevent high variance

        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)  # Increase beta for future sampling
        return batch, idxs, is_weights  # Return the sampled experiences, their indices, and importance-sampling weights

    def update(self, idx, error):
        # Update the priority of a specific experience
        priority = (abs(error) + self.epsilon) ** self.alpha  # Compute new priority
        self.tree.update(idx, priority)  # Update the SumTree with the new priority
