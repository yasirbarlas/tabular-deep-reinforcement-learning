import operator
import random
import numpy as np
from collections import deque

class SegmentTree:
    """
    Create SegmentTree.
    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py
    Attributes:
        capacity (int)
        tree (list)
        operation (function)
    """
    def __init__(self, capacity, operation, init_value):
        """
        Initialisation.
        Args:
            capacity (int)
            operation (function)
            init_value (float)
        """
        assert (capacity > 0 and capacity & (capacity - 1) == 0), "capacity must be positive and a power of 2."
        self.capacity = capacity
        self.tree = [init_value for _ in range(2 * capacity)]
        self.operation = operation

    def _operate_helper(self, start, end, node, node_start, node_end):
        """
        Returns result of operation in segment.
        """
        if start == node_start and end == node_end:
            return self.tree[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._operate_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._operate_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self.operation(self._operate_helper(start, mid, 2 * node, node_start, mid), self._operate_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end))

    def operate(self, start = 0, end = 0):
        """
        Returns result of applying 'self.operation'.
        """
        if end <= 0:
            end += self.capacity
        end -= 1

        return self._operate_helper(start, end, 1, 0, self.capacity - 1)

    def __setitem__(self, idx, val):
        """
        Set value in tree.
        """
        idx += self.capacity
        self.tree[idx] = val

        idx //= 2
        while idx >= 1:
            self.tree[idx] = self.operation(self.tree[2 * idx], self.tree[2 * idx + 1])
            idx //= 2

    def __getitem__(self, idx):
        """
        Get real value in leaf node of tree.
        """
        assert 0 <= idx < self.capacity

        return self.tree[self.capacity + idx]

class SumSegmentTree(SegmentTree):
    """
    Create SumSegmentTree.
    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py
    """
    def __init__(self, capacity):
        """Initialisation.
        Args:
            capacity (int)
        """
        super(SumSegmentTree, self).__init__(capacity = capacity, operation = operator.add, init_value = 0.0)

    def sum(self, start = 0, end = 0):
        """Returns arr[start] + ... + arr[end]."""
        return super(SumSegmentTree, self).operate(start, end)

    def retrieve(self, upperbound):
        """
        Find the highest index `i` about upper bound in the tree
        """
        assert 0 <= upperbound <= self.sum() + 1e-5, "upperbound: {}".format(upperbound)

        idx = 1

        while idx < self.capacity:  # while non-leaf
            left = 2 * idx
            right = left + 1
            if self.tree[left] > upperbound:
                idx = 2 * idx
            else:
                upperbound -= self.tree[left]
                idx = right
        return idx - self.capacity

class MinSegmentTree(SegmentTree):
    """Create SegmentTree.
    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py
    """
    def __init__(self, capacity):
        """
        Initialisation.

        Args:
            capacity (int)
        """
        super(MinSegmentTree, self).__init__(capacity = capacity, operation = min, init_value = float("inf"))

    def min(self, start = 0, end = 0):
        """Returns min(arr[start], ...,  arr[end])."""
        return super(MinSegmentTree, self).operate(start, end)

class ReplayBuffer:
    """
    A simple NumPy replay buffer.
    """
    def __init__(self, obs_shape, size, batch_size = 32):
        self.obs_buf = np.zeros([size] + list(obs_shape), dtype = np.float32)
        self.next_obs_buf = np.zeros([size] + list(obs_shape), dtype = np.float32)
        self.acts_buf = np.zeros([size], dtype = np.float32)
        self.rews_buf = np.zeros([size], dtype = np.float32)
        self.done_buf = np.zeros(size, dtype = np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self):
        idxs = np.random.choice(self.size, size = self.batch_size, replace = False)

        return dict(obs = self.obs_buf[idxs], next_obs = self.next_obs_buf[idxs], acts = self.acts_buf[idxs], rews = self.rews_buf[idxs], done = self.done_buf[idxs], indices = idxs)

    def __len__(self):
        return self.size

class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Replay Buffer.
    """
    def __init__(self, obs_shape, size, batch_size = 32, alpha = 0.5):
        """Initialisation."""
        assert alpha >= 0

        super(PrioritizedReplayBuffer, self).__init__(obs_shape, size, batch_size)
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha

        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def store(self, obs, act, rew, next_obs, done):
        """
        Store experience and priority.
        """
        super().store(obs, act, rew, next_obs, done)

        self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.max_size

    def sample_batch(self, beta = 0.4):
        """
        Sample a batch of experiences.
        """
        assert len(self) >= self.batch_size
        assert beta > 0

        indices = self._sample_proportional()

        obs = self.obs_buf[indices]
        next_obs = self.next_obs_buf[indices]
        acts = self.acts_buf[indices]
        rews = self.rews_buf[indices]
        done = self.done_buf[indices]
        weights = np.array([self._calculate_weight(i, beta) for i in indices])

        return dict(obs = obs, next_obs = next_obs, acts = acts, rews = rews, done = done, weights = weights, indices = indices)

    def update_priorities(self, indices, priorities):
        """
        Update priorities of sampled transitions.
        """
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)

    def _sample_proportional(self):
        """
        Sample indices based on proportions.
        """
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / self.batch_size

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)

        return indices

    def _calculate_weight(self, idx, beta):
        """
        Calculate the weight of the experience at idx.
        """
        # Get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)

        # Calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight

        return weight