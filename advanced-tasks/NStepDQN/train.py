import numpy as np

from agent import *
from wrappers import *

import gymnasium as gym
from gymnasium.wrappers import FrameStack, TransformReward
from vizdoom import gymnasium_wrapper

##### PARAMETERS #####
env_id = "VizdoomDefendCenter-v0"
num_frames = 1000000
memory_size = 100000
batch_size = 32
target_update = 8000
seed = 50
## Discount Factor ##
gamma = 0.99
## Prioritised Experience Replay ##
alpha = 0.5
beta = 0.4
prior_eps = 0.000001
## Noisy Network ##
noisy_std = 0.5
## N-step Returns ##
n_step = 3
## Optimizer ##
learning_rate = 0.00001
optimizer = "adam"

def main():
    # Set random seed for reproducibility
    randomer = 50
    random.seed(randomer)
    np.random.seed(randomer)
    torch.manual_seed(randomer)
    torch.cuda.manual_seed_all(randomer)

    # Environment
    env = gym.make(env_id, render_mode = "rgb_array", frame_skip = 4)

    # Apply Wrappers to environment (working for ViZDoom environments only, replace with standard Gymnasium ones for other environments)
    env = ObservationWrapper(env)
    env = GrayScaleObservation(env)
    # Transforms reward by scalar, can be commented out if not needed
    #env = TransformReward(env, lambda r: r * 0.01)

    # Frame stacking
    if gym.__version__ < "0.26":
        env = FrameStack(env, num_stack = 4, new_step_api = True)
    else:
        env = FrameStack(env, num_stack = 4)

    agent = NStepAgent(env, memory_size, batch_size, target_update, seed, gamma, alpha, beta, prior_eps, noisy_std, n_step, learning_rate, optimizer)

    agent.train(num_frames)

    agent.test()

if __name__ == "__main__":
    main()