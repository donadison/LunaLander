import numpy as np


class SumTree:
    # A SumTree data structure that maintains a binary tree for prioritized sampling.
    def __init__(self, capacity):
        self.capacity = capacity  # Maximum capacity of the SumTree
        self.tree = np.zeros(2 * capacity - 1)  # Initialize tree with zeros, size is 2 * capacity - 1
        self.data = [None] * capacity  # Store experiences/data in a list of length 'capacity'
        self.write = 0  # Index to write new data
        self.current_size = 0  # Tracks the number of elements currently stored

    def add(self, priority, data):
        # Add a new experience to the SumTree with its priority
        tree_idx = self.write + self.capacity - 1  # Calculate tree index for the leaf node
        self.data[self.write] = data  # Store the data at the appropriate position
        self.update(tree_idx, priority)  # Update the tree with the given priority

        self.write = (self.write + 1) % self.capacity  # Wrap-around for circular buffer
        self.current_size = min(self.current_size + 1, self.capacity)  # Update current size

    def update(self, tree_idx, priority):
        # Update the priority at a specific tree index and propagate the changes upwards
        change = priority - self.tree[tree_idx]  # Calculate the change in priority
        if change == 0:
            return  # Skip unnecessary updates if priority is the same

        self.tree[tree_idx] = priority  # Update the priority at the tree index
        idx = tree_idx

        # Propagate the change upwards to the parent nodes
        while idx > 0:
            idx = (idx - 1) // 2  # Move to parent node
            self.tree[idx] += change  # Update the parent node

    def get_leaf(self, value):
        # Retrieve a leaf node based on a given value, which is randomly sampled
        parent_idx = 0  # Start from the root of the tree

        while True:
            left_child = 2 * parent_idx + 1  # Left child index
            right_child = left_child + 1  # Right child index

            if left_child >= len(self.tree):  # If we reach a leaf node
                leaf_idx = parent_idx
                break
            else:
                if value <= self.tree[left_child]:
                    parent_idx = left_child  # Go to the left child
                else:
                    value -= self.tree[left_child]  # Subtract the left child's priority
                    parent_idx = right_child  # Go to the right child

        data_idx = leaf_idx - self.capacity + 1  # Calculate the index of the stored data
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]  # Return leaf index, priority, and data

    def total_priority(self):
        # Return the total priority from the root node
        return self.tree[0]  # Root node contains the sum of all priorities

    def __len__(self):
        # Overload for the return the current size of the SumTree
        return self.current_size