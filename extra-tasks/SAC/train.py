import numpy as np
import random

from agent import *
from wrappers import *

import gymnasium as gym

##### PARAMETERS #####
env_id = "Ant-v4"
num_frames = 5000000
memory_size = 300000
batch_size = 128
seed = 50
## SAC ##
hidden_dim = 256
tau = 0.005
initial_random_steps = 1000
updates_per_step = 1
target_update_interval = 1
## Discount Factor ##
gamma = 0.99
## Optimizer ##
learning_rate = 0.0003
optimizer = "adam"

def main():
    # Set random seed for reproducibility
    randomer = 50
    random.seed(randomer)
    np.random.seed(randomer)
    torch.manual_seed(randomer)
    torch.cuda.manual_seed_all(randomer)

    # Environment
    env = gym.make(env_id, render_mode = "rgb_array")
    # Enable if desired
    #env = ActionNormalizer(env)
    
    agent = SACAgent(env, memory_size, batch_size, seed, gamma, tau, initial_random_steps, updates_per_step, target_update_interval, hidden_dim, learning_rate, optimizer)

    #agent._load_checkpoint("checkpoints/checkpoint_sac_latest.pth.tar", include_optimiser = True)
    
    agent.train(num_frames)

    agent.test()

if __name__ == "__main__":
    main()
