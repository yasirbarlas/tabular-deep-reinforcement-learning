# INM707: Deep Reinforcement Learning

Note: This repository and its contents support the coursework of the INM707 module at City, University of London.
Note for examiners: The Advanced Individual Tasks and Extra Task were run on the Hyperion High-Performance Computer, hence the lack of Jupyter Notebooks.

The ``requirements.txt`` files assume that you have PyTorch (ideally the CUDA version for GPU usage) already installed. Find more information about how to install PyTorch [here](https://pytorch.org/get-started/locally/).

## Basic Tasks

We create a Gridworld-like environment using NumPy and Matplotlib. The environment specifically looks at an agent traversing a single-story house (or apartment), with the aim of cleaning the carpets of the house. We refer to the agent as a robot vacuum cleaner. The agent receives a positive reward for cleaning carpets, a negative reward for bumping into walls/furniture or trying to escape the grid, a negative reward for moving onto states with cables/wires, and lastly a positive reward for entering the terminal state (a charging station).

The relevant Jupyter Notebook can be found in the [basic-tasks](../main/basic-tasks) folder.

## Advanced Tasks

We investigate the [(Target Network + Experience Replay) Deep Q-Network (DQN)](https://www.nature.com/articles/nature14236), the [Prioritised Experience Replay DQN](https://arxiv.org/abs/1511.05952v4), the [Noisy Network DQN](https://arxiv.org/abs/1706.10295), and the [N-Step Return DQN](https://link.springer.com/article/10.1007/BF00115009) algorithms on the [ViZDoom 'Defend The Center'](https://vizdoom.farama.org/) environment. Each algorithm utilises the previous ones sequentially, similar to what is done in [Rainbow DQN](https://arxiv.org/abs/1710.02298). In other words, we look at a Target Network + Experience Replay DQN, a Target Network + Prioritised Experience Replay DQN, a Target Network + Prioritised Experience Replay + Noisy Network DQN, and lastly a Target Network + Prioritised Experience Replay + Noisy Network + N-Step Return DQN. We provide the code in standard .py form and as Jupyter Notebooks, and neither of these depend on each other to run.

We use Python 3.10.13, PyTorch (with CUDA 12.1), Gymnasium 0.29.0, ViZDoom 1.2.3, and for recording the agent in the testing phase, MoviePy 1.0.3. Instructions for installing ViZDoom (through Github or pip) can be found [here](https://vizdoom.farama.org/).

Much work on reinforcement learning seems to be built using the old versions of PyTorch and Gym. Here, we ensure that our code runs well on the latest versions of these libraries.

The relevant Jupyter Notebooks and code can be found in the [advanced-tasks](../main/advanced-tasks) folder.

We provide model checkpoints for each of the DQN algorithms we investigate, all of which can be downloaded [here](https://cityuni-my.sharepoint.com/:f:/g/personal/yasir-zubayr_barlas_city_ac_uk/EsKQGEAsns9BgfrswKtlM9QBndWCBJz82Eum_YV1W5zLOg?e=DW3nc7).

The following is information regarding the ViZDoom environment we use:

```
DEFEND THE CENTER
The purpose of this scenario is to teach the agent that killing the monsters is GOOD and when monsters kill you is BAD. In addition, wasting ammunition is not very good either. Agent is rewarded only for killing monsters so he has to figure out the rest for himself.

The map is a large circle. A player is spawned in the exact center. 5 melee-only, monsters are spawned along the wall. Monsters are killed after a single shot. After dying, each monster is respawned after some time. The episode ends when the player dies (it’s inevitable because of limited ammo).

REWARDS:

+1 for killing a monster

-1 for death

CONFIGURATION:

3 available buttons: turn left, turn right, shoot (attack)

2 available game variables: player’s health and ammo

timeout = 2100

difficulty level (doom_skill) = 3

Gymnasium/Gym id: "VizdoomDefendCenter-v0"
```

We note that our code is inspired by the [Rainbow is all you need!](https://github.com/Curt-Park/rainbow-is-all-you-need) repository, and that while much of our code is similar, we make certain adjustments to be suited for the ViZDoom environment, and we make the sequential improvements to each algorithm we examine as mentioned previously. Since we are working with pixel data, we use the convolutional neural networks as found in the [RL Adventure](https://github.com/higgsfield/RL-Adventure/) repository.

## Individual Advanced Tasks

We use the Proximal Policy Optimisation algorithm on the multi-agent [PettingZoo Atari 'Space Invaders'](https://pettingzoo.farama.org/environments/atari/space_invaders/) environment, which is an on-policy policy gradient algorithm that tries to make large policy improvements without causing performance collapse. Rather than implementing the algorithm from scratch, we utilise the Ray RLlib library instead. We provide the code in standard .py form.

Our algorithms do not use the latest Ray RLlib API stack, but do use the latest version of the old API stack. We recommend installing the "Daily Release" of Ray RLlib rather than the latest official release using pip, though both installation methods should work. Find more information on the [Ray](https://docs.ray.io/en/latest/ray-overview/installation.html) website.

We use Python 3.10.13, Gymnasium 0.28.1 (Ray RLlib does not support the latest version), Ray 3.0.0.dev0 (Daily Release), PyTorch (with CUDA 12.1), PettingZoo 1.24.3, and SuperSuit 3.9.2. The ROMs for the Atari environments also need to be installed, which can be done through: ```pip install autorom[accept-rom-license]```. See the [AutoROM](https://github.com/Farama-Foundation/AutoROM) repository for more information.

The relevant code can be found in the [advanced-tasks-individual](../main/advanced-tasks-individual) folder.

We provide model checkpoints and our final results for each of the experiments conducted, all of which can be downloaded [here](https://cityuni-my.sharepoint.com/:f:/g/personal/yasir-zubayr_barlas_city_ac_uk/EiBjxuscMF1OuP81VFB2RhgBhSTa1fwDOjwgaNJ6llUHUg?e=bqYcVa).

We note that our code is based on a [PettingZoo tutorial](https://pettingzoo.farama.org/tutorials/rllib/pistonball/), which has been adapted to work on our environment.

## Extra Tasks

The Soft Actor-Critic algorithm is considered a state-of-the-art algorithm, notably in continuous control tasks. It combines actor-critic methods with maximum entropy reinforcement learning, encouraging exploration while simultaneously optimising for both policy improvement and entropy maximisation. We provide code to work on many of the continuous control tasks found on [Gymnasium](https://gymnasium.farama.org/index.html). We provide the code in standard .py form.

We use Python 3.10.13, PyTorch (with CUDA 12.1), Gymnasium 0.29.0, and for recording the agent in the testing phase, MoviePy 1.0.3.

Much work on reinforcement learning seems to be built using the old versions of PyTorch and Gym. Here, we ensure that our code runs well on the latest versions of these libraries.

The relevant code can be found in the [extra-tasks](../main/extra-tasks) folder.

We provide model checkpoints for each of the experiments conducted, all of which can be downloaded [here](https://cityuni-my.sharepoint.com/:f:/g/personal/yasir-zubayr_barlas_city_ac_uk/EiwQl3vYcRVKvJxY9MreegMBntqf0vxUfaEaM9z7WztAQQ?e=djqEYW).

We note that our Agent class is inspired by the [PG is all you need!](https://github.com/MrSyee/pg-is-all-you-need) repository, and much of the code is adapted from the [pytorch-soft-actor-critic](https://github.com/pranz24/pytorch-soft-actor-critic) repository.
