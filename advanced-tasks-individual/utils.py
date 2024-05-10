from torch import nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import supersuit as ss
from pettingzoo.atari import space_invaders_v2 as environment_ma

## As found in https://pettingzoo.farama.org/tutorials/rllib/pistonball/ ##
## With some changes ##

# Define a custom CNN model for the PPO algorithm
class CNNModelV2(TorchModelV2, nn.Module):
    def __init__(self, obs_space, act_space, num_outputs, *args, **kwargs):
        TorchModelV2.__init__(self, obs_space, act_space, num_outputs, *args, **kwargs)
        nn.Module.__init__(self)
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, [8, 8], stride = (4, 4)),
            nn.ReLU(),
            nn.Conv2d(32, 64, [4, 4], stride = (2, 2)),
            nn.ReLU(),
            nn.Conv2d(64, 64, [3, 3], stride = (1, 1)),
            nn.ReLU(),
            nn.Flatten(),
            (nn.Linear(3136, 512)),
            nn.ReLU(),
        )
        self.policy_fn = nn.Linear(512, num_outputs)
        self.value_fn = nn.Linear(512, 1)

    def forward(self, input_dict, state, seq_lens):
        model_out = self.model(input_dict["obs"].permute(0, 3, 1, 2))
        self._value_out = self.value_fn(model_out)
        return self.policy_fn(model_out), state

    def value_function(self):
        return self._value_out.flatten()

# Define an environment creator function
def env_creator(args):
    env = environment_ma.parallel_env()
    env = ss.color_reduction_v0(env, mode = "B")
    env = ss.dtype_v0(env, "float32")
    env = ss.resize_v1(env, x_size = 84, y_size = 84)
    env = ss.normalize_obs_v0(env, env_min = 0, env_max = 1)
    env = ss.frame_stack_v1(env, 4)
    return env