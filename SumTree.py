import numpy as np

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # SumTree structure
        self.data = [None] * capacity  # Use a list for data
        self.write = 0
        self.current_size = 0  # Track current number of elements

    def add(self, priority, data):
        tree_idx = self.write + self.capacity - 1
        self.data[self.write] = data  # Store the experience
        self.update(tree_idx, priority)

        self.write = (self.write + 1) % self.capacity
        self.current_size = min(self.current_size + 1, self.capacity)

    def update(self, tree_idx, priority):
        change = priority - self.tree[tree_idx]
        if change == 0:
            return  # Skip unnecessary updates

        self.tree[tree_idx] = priority
        idx = tree_idx

        # Propagate changes upwards
        while idx > 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def get_leaf(self, value):
        parent_idx = 0

        while True:
            left_child = 2 * parent_idx + 1
            right_child = left_child + 1

            if left_child >= len(self.tree):  # Reached a leaf node
                leaf_idx = parent_idx
                break
            else:
                if value <= self.tree[left_child]:
                    parent_idx = left_child
                else:
                    value -= self.tree[left_child]
                    parent_idx = right_child

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    def total_priority(self):
        return self.tree[0]  # Root node

    def __len__(self):
        return self.current_size