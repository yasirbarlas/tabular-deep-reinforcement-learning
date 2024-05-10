import os
import pickle
import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.optim as optim
import torch.nn.functional as F

from gymnasium.wrappers import RecordVideo

from buffers import *
from models import *
from utils import *

## As found in https://github.com/Curt-Park/rainbow-is-all-you-need/blob/master/05.noisy_net.ipynb ##
## With some changes ##

class NoisyAgent:
    """
    Noisy Network (with PER) DQN Agent
    """

    def __init__(self, env, memory_size, batch_size, target_update, seed, gamma = 0.99, alpha = 0.5, beta = 0.4, prior_eps = 0.000001, noisy_std = 0.5, learning_rate = 0.0000625, optimizer = "adam"):
        """
        Initialise the Noisy agent with the provided parameters.
        
        Args:
            env (gymnasium.Env): Gymnasium environment
            memory_size: Size of the replay buffer
            batch_size: Batch size for training
            target_update: Interval for target network update
            seed (int): Seed for random number generation
            gamma: Discount factor (default: 0.99)
            alpha: Prioritization exponent (default: 0.5)
            beta: Importance sampling exponent (default: 0.4)
            prior_eps: Small constant to avoid division by zero (default: 0.000001)
            noisy_std: Standard deviation of the noisy network (default: 0.5)
            learning_rate: Learning rate for the optimizer (default: 0.0000625)
            optimizer (str): Name of the optimizer to use ("adam", "rmsprop", or "sgd") (default: "adam")
        """
        obs_dim = env.observation_space.shape
        action_dim = env.action_space.n

        self.env = env
        self.batch_size = batch_size
        self.target_update = target_update
        self.seed = seed
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.prior_eps = prior_eps
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        
        # Prioritised Experience Replay
        self.memory = PrioritizedReplayBuffer(obs_dim, memory_size, batch_size, alpha)
        
        # Action space seed
        self.env.action_space.seed(seed)

        # Neural network models setup
        self.dqn = NoisyDQN(obs_dim, action_dim, noisy_std = noisy_std).to(self.device)
        self.dqn_target = NoisyDQN(obs_dim, action_dim, noisy_std = noisy_std).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()
        
        # Optimizer setup
        optimizer.lower()
        if optimizer == "adam":
            self.optimizer = optim.Adam(self.dqn.parameters(), lr = learning_rate)
        if optimizer == "rmsprop":
            self.optimizer = optim.RMSprop(self.dqn.parameters(), lr = learning_rate)
        if optimizer == "sgd":
            self.optimizer = optim.SGD(self.dqn.parameters(), lr = learning_rate)

        # Transition to store in memory
        self.transition = list()
        
        # Mode: train / test
        self.is_test = False

    def select_action(self, state):
        """
        Select an action from the input state.
        """
        selected_action = self.dqn.act(state)
        
        if not self.is_test:
            self.transition = [state, selected_action]
        
        return selected_action

    def step(self, action):
        """
        Take an action and return the response of the environment.
        """
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated

        if not self.is_test:
            self.transition += [reward, next_state, done]
            self.memory.store(*self.transition)
    
        return next_state, reward, done

    def update_model(self):
        """
        Update the model by gradient descent.
        """
        # PER needs beta to calculate weights
        samples = self.memory.sample_batch(self.beta)
        weights = torch.tensor(samples["weights"].reshape(-1, 1), dtype = torch.float, device = self.device).to(self.device)
        indices = samples["indices"]

        # PER: importance sampling before average
        elementwise_loss = self._compute_dqn_loss(samples)
        loss = torch.mean(elementwise_loss * weights)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)

        # NoisyNet: reset noise
        self.dqn.reset_noise()
        self.dqn_target.reset_noise()

        return loss.item()
        
    def train(self, num_frames, plotting_interval = 1000):
        """
        Train the agent.
        """
        self.is_test = False
        
        state, _ = self.env.reset(seed = self.seed)
        update_cnt = 0
        losses = []
        scores = []
        score = 0

        for frame_idx in range(1, num_frames + 1):
            # At the halfway point, reduce learning rate by a tenth
            if frame_idx == int(num_frames // 2):
                adjust_learning_rate(self.optimizer, 0.1)

            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

            # PER: increase beta
            fraction = min(frame_idx / num_frames, 1.0)
            self.beta = self.beta + fraction * (1.0 - self.beta)

            # If episode ends
            if done:
                state, _ = self.env.reset(seed = self.seed)
                scores.append(score)
                score = 0

            # If training is ready
            if len(self.memory) >= self.batch_size:
                loss = self.update_model()
                losses.append(loss)
                update_cnt += 1

                # If hard update is needed
                if update_cnt % self.target_update == 0:
                    self._target_hard_update()

            # Plot and save results, and save models
            if frame_idx % plotting_interval == 0:
                self._plot(scores, losses)
                self._save(scores, losses)

                print(f"Frame: {frame_idx}, Mean of last 10 rewards: {np.mean(scores[-10:])}")

                # Create checkpoint folder
                if not os.path.exists("checkpoints"):
                    os.makedirs("checkpoints")

                # Save the model checkpoint
                checkpoint_name = f"checkpoint_noisy_dqn_latest.pth.tar"
                checkpoint_path = os.path.join("checkpoints", checkpoint_name)
                torch.save({"current_model": self.dqn.state_dict(), "target_model": self.dqn_target.state_dict(), "optimizer": self.optimizer.state_dict(), "losses": losses, "rewards": scores, "frame": frame_idx}, checkpoint_path)

        print("Training successfully completed.")

        self.env.close()

    def test(self, video_folder = "noisy-dqn_agent_video"):
        """
        Test the agent.
        """
        self.is_test = True
        
        # Create checkpoint folder
        if not os.path.exists(video_folder):
            os.makedirs(video_folder)
        
        # For recording a video of agent
        naive_env = self.env
        self.env = RecordVideo(self.env, video_folder = video_folder)
        
        state, _ = self.env.reset(seed = self.seed)
        done = False
        score = 0
        
        while not done:
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward
        
        print("Score: ", score)
        self.env.close()
        
        # Reset
        self.env = naive_env

    def _compute_dqn_loss(self, samples):
        """
        Return DQN loss.
        """
        state = torch.tensor(samples["obs"], dtype = torch.float, device = self.device).to(self.device)
        next_state = torch.tensor(samples["next_obs"], dtype = torch.float, device = self.device).to(self.device)
        action = torch.tensor(samples["acts"].reshape(-1, 1), dtype = torch.long, device = self.device).to(self.device)
        reward = torch.tensor(samples["rews"].reshape(-1, 1), dtype = torch.float, device = self.device).to(self.device)
        done = torch.tensor(samples["done"].reshape(-1, 1), dtype = torch.float, device = self.device).to(self.device)

        curr_q_value = self.dqn(state).gather(1, action)
        next_q_value = self.dqn_target(next_state).max(dim = 1, keepdim = True)[0].detach()
        mask = 1 - done
        target = (reward + self.gamma * next_q_value * mask).to(self.device)

        # Calculate DQN loss
        loss = F.smooth_l1_loss(curr_q_value, target, reduction = "none")

        return loss

    def _target_hard_update(self):
        """
        Hard update: target <- local.
        """
        self.dqn_target.load_state_dict(self.dqn.state_dict())

    def _save(self, rewards, losses):
        """
        Save training results to a pickle file.
        """
        # Save results to a file
        with open("noisy-dqn-results.pkl", "wb") as f:
            pickle.dump(rewards, f)
            pickle.dump(losses, f)

    def _plot(self, rewards, losses, moving_average_window = 100):
        """
        Plot training curves.
        """
        plt.figure(figsize = (40, 6))
        
        # Combined plot of rewards, moving average, and loss
        plt.subplot(131)
        plt.title("Noisy-DQN Rewards Per Episode")
        plt.plot(rewards, label = "Reward")
        if len(rewards) >= moving_average_window:
            plt.plot(moving_average(rewards, moving_average_window), label = "Moving Average", color = "red")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        
        plt.subplot(132)
        plt.title("Noisy-DQN Loss Per Frame")
        plt.plot(losses, label = "DQN Loss")
        plt.xlabel("Frame")
        plt.ylabel("Loss")
        
        plt.savefig("noisy_dqn_plot.pdf")
        plt.close()

    def _load_checkpoint(self, checkpoint_path, include_optimiser = True):
        model = torch.load(checkpoint_path, map_location = self.device)
        self.dqn.load_state_dict(model["current_model"])
        self.dqn_target.load_state_dict(model["target_model"])
        
        if include_optimiser == True:
            self.optimizer.load_state_dict(model["optimizer"])