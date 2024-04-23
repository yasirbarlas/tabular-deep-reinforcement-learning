import numpy as np

## As found in https://github.com/Curt-Park/rainbow-is-all-you-need/blob/master/01.dqn.ipynb ##

class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(self, obs_shape, size, batch_size = 32, n_step = 3, gamma = 0.99):
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
