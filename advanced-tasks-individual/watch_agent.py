import argparse
import os

import ray
from PIL import Image
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from utils import CNNModelV2, env_creator

## As found in https://pettingzoo.farama.org/tutorials/rllib/pistonball/ ##
## With some changes ##

if __name__ == "__main__":
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    parser = argparse.ArgumentParser(
        description = "Render pretrained policy loaded from checkpoint."
    )
    parser.add_argument(
        "--checkpoint-path",
        help = "Path to the checkpoint. This path will likely be something like this: `~/ray_results/pistonball_v6/PPO/PPO_pistonball_v6_660ce_00000_0_2021-06-11_12-30-57/checkpoint_000050/checkpoint-50`",
    )

    parser.add_argument(
        "--filenamesave",
        help = "Name of file to save as .gif.",
    )

    args = parser.parse_args()

    if args.checkpoint_path is None:
        print("The following arguments are required: --checkpoint-path, --filenamesave")
        exit(0)

    if args.filenamesave is None:
        print("The following arguments are required: --checkpoint-path, --filenamesave")
        exit(0)

    checkpoint_path = os.path.expanduser(args.checkpoint_path)
    filenamesave = str(args.filenamesave)

    # Create environment
    env = env_creator()

    # Set environment name
    env_name = "space_invaders_v2"

    # Register the environment creator function
    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator()))
    # Register the custom CNN model
    ModelCatalog.register_custom_model("CNNModelV2", CNNModelV2)

    # Initialise Ray
    ray.init()

    # Get checkpoint
    PPOagent = PPO.from_checkpoint(checkpoint_path)

    reward_sum = 0
    frame_list = []
    i = 0
    env.reset()

    # Iterate over every agent
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        reward_sum += reward
        if termination or truncation:
            action = None
        else:
            action = PPOagent.compute_single_action(observation)

        env.step(action)
        i += 1
        if i % (len(env.possible_agents) + 1) == 0:
            img = Image.fromarray(env.render())
            frame_list.append(img)
    env.close()

    print(reward_sum)
    frame_list[0].save(
        f"{filenamesave}.gif", save_all = True, append_images = frame_list[1:], duration = 3, loop = 0
    )