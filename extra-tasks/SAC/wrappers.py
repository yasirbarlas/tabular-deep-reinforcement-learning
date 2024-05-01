import gymnasium as gym
import numpy as np

## As found in https://github.com/MrSyee/pg-is-all-you-need/blob/master/05.SAC.ipynb ##
## Not necessary to use, and is not used in our experiments ##

class ActionNormalizer(gym.ActionWrapper):
    """
    Rescale and relocate the actions.
    """

    def action(self, action):
        """
        Change the range (-1, 1) to (low, high).
        """
        low = self.action_space.low
        high = self.action_space.high

        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor

        action = action * scale_factor + reloc_factor
        action = np.clip(action, low, high)

        return action

    def reverse_action(self, action):
        """
        Change the range (low, high) to (-1, 1).
        """
        low = self.action_space.low
        high = self.action_space.high

        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor

        action = (action - reloc_factor) / scale_factor
        action = np.clip(action, -1.0, 1.0)

        return action