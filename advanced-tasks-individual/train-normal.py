#### Normal PPO algorithm ####

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from utils import CNNModelV2, env_creator

## As found in https://pettingzoo.farama.org/tutorials/rllib/pistonball/ ##
## With some changes ##

if __name__ == "__main__":
    # Initialise Ray
    ray.init()

    # Set environment name
    env_name = "space_invaders_v2"

    # Register the environment creator function
    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))
    # Register the custom CNN model
    ModelCatalog.register_custom_model("CNNModelV2", CNNModelV2)

    # Parameters not listed are set to their default state
    # Configure PPO algorithm
    config = (
        PPOConfig()
        .environment(env = env_name, clip_actions = True)
        .rollouts(num_rollout_workers = 4, rollout_fragment_length = 128)
        .training(
            train_batch_size = 512,
            lr = 0.00002,
            gamma = 0.99,
            lambda_ = 0.9,
            use_gae = True,
            use_kl_loss = True,
            clip_param = 0.4,
            grad_clip = None,
            entropy_coeff = 0.1,
            vf_loss_coeff = 0.25,
            sgd_minibatch_size = 64,
            num_sgd_iter = 10,
        )
        .debugging(log_level = "ERROR")
        .framework(framework = "torch") # Change framework as desired
        .resources(num_gpus = 1) # Change number of GPUs as desired
    )

    # Run the PPO algorithm with Ray Tune, creates a folder with results/checkpoints in directory 'local_dir' with folder as 'name'
    tune.run(
        "PPO",
        name = f"PPO_{env_name}_multi-agent_normal",
        stop = {"timesteps_total": 5000000},
        checkpoint_freq = 100,
        local_dir = "~/ray_results/" + env_name,
        config = config.to_dict(),
    )