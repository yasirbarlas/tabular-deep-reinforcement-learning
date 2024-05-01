import gymnasium as gym
import cv2
import numpy as np

class ObservationWrapper(gym.ObservationWrapper):
    """
    ViZDoom environments return dictionaries as observations, containing
    the main image as well other info.
    The image is also too large for normal training.

    This wrapper replaces the dictionary observation space with a simple
    Box space (i.e., only the RGB image), and also resizes the image to a
    smaller size.

    NOTE: Ideally, you should set the image size to smaller in the scenario files
          for faster running of ViZDoom. This can really impact performance,
          and this code is pretty slow because of this!
    """
    def __init__(self, env, shape = (84, 84)):
        super().__init__(env)
        self.image_shape = shape
        self.image_shape_reverse = shape[::-1]

        # Create new observation space with the new shape
        print(env.observation_space)
        num_channels = env.observation_space["screen"].shape[-1]
        new_shape = (shape[0], shape[1], num_channels)
        self.observation_space = gym.spaces.Box(
            0, 255, shape = new_shape, dtype = np.uint8
        )

    def observation(self, observation):
        observation = cv2.resize(observation["screen"], self.image_shape_reverse)
        return observation

class GrayScaleObservation(gym.ObservationWrapper):
    """
    Convert the image observation from RGB to grayscale.
    Example:
        >>> env = gym.make('CarRacing-v1')
        >>> env.observation_space
        Box(0, 255, (96, 96, 3), uint8)
        >>> env = GrayScaleObservation(gym.make('CarRacing-v1'))
        >>> env.observation_space
        Box(0, 255, (96, 96), uint8)
        >>> env = GrayScaleObservation(gym.make('CarRacing-v1'), keep_dim=True)
        >>> env.observation_space
        Box(0, 255, (96, 96, 1), uint8)
    """
    def __init__(self, env: gym.Env, keep_dim: bool = False):
        """
        Convert the image observation from RGB to grayscale.
        Args:
            env (Env): The environment to apply the wrapper
            keep_dim (bool): If `True`, a singleton dimension will be added, i.e. observations are of the shape AxBx1.
                Otherwise, they are of shape AxB.
        """
        super().__init__(env)
        self.keep_dim = keep_dim

        assert (
            isinstance(self.observation_space, gym.spaces.Box)
            and len(self.observation_space.shape) == 3
            and self.observation_space.shape[-1] == 3
        )

        obs_shape = self.observation_space.shape[:2]
        if self.keep_dim:
            self.observation_space = gym.spaces.Box(
                low = 0, high = 255, shape = (obs_shape[0], obs_shape[1], 1), dtype = np.uint8
            )
        else:
            self.observation_space = gym.spaces.Box(
                low = 0, high = 255, shape = obs_shape, dtype = np.uint8
            )

    def observation(self, observation):
        """
        Converts the colour observation to grayscale.
        Args:
            observation: Color observations
        Returns:
            Grayscale observations
        """

        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        if self.keep_dim:
            observation = np.expand_dims(observation, -1)
        return observation