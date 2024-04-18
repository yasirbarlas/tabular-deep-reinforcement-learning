# INM707: Deep Reinforcement Learning

Note: This repository and its contents support the coursework of the INM707 module at City, University of London.

## Basic Tasks

We create a Gridworld-like environment using NumPy and Matplotlib. The environment specifically looks at an agent traversing a single-story house (or apartment), with the aim of cleaning the carpets of the house. We refer to the agent as a robot vacuum cleaner. The agent receives a positive reward for cleaning carpets, a negative reward for bumping into walls/furniture or trying to escape the grid, a negative reward for moving onto states with cables/wires, and lastly a positive reward for entering the terminal state (a charging station).

The relevant Jupyter Notebook can be found in the [basic-tasks](../main/basic-tasks) folder.

## Advanced Tasks

We investigate the [Deep Q-Network (DQN)](https://arxiv.org/abs/1312.5602), [Rainbow DQN](https://arxiv.org/abs/1710.02298), and [Quantile Regression DQN](https://arxiv.org/abs/1710.10044) algorithms on the ViZDoom 'Defend The Center' environment. We provide the code in standard .py form and as Jupyter Notebooks, and neither of these depend on each other to run.

We use Python 3.10.13, PyTorch (with CUDA 12.1), Gymnasium 0.29.0, and ViZDoom 1.2.3 here. Instructions for installing ViZDoom (through Github or pip) can be found [here](https://vizdoom.farama.org/).

Much work on reinforcement learning seems to be built using the old versions of PyTorch and Gym. Here, we ensure that our code runs well on the latest versions of these libraries.

The relevant Jupyter Notebooks and code can be found in the [advanced-tasks](../main/advanced-tasks) folder.

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

## Individual Advanced Tasks

## Extra Tasks
