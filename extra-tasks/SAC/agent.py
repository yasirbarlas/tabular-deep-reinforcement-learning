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

## Agent class is inspired by https://github.com/MrSyee/pg-is-all-you-need/blob/master/05.SAC.ipynb ##
## Code for SAC adapted from https://github.com/pranz24/pytorch-soft-actor-critic/blob/master/sac.py ##
## With some changes/updates ##

class SACAgent:
    """
    SAC Agent
    """
    def __init__(self, env, memory_size, batch_size, seed, gamma = 0.99, tau = 0.005, initial_random_steps = 10000, updates_per_step = 1, target_update_interval = 1, hidden_dim = 128, learning_rate = 0.0001, optimizer = "adam"):
        """
        Initialise the SAC agent with the provided parameters.

        Args:
            env (gymnasium.Env): Gymnasium environment
            memory_size (int): Size of the replay memory
            batch_size (int): Batch size for sampling from the replay memory
            seed (int): Seed for random number generation
            gamma (float): Discount factor (default: 0.99)
            tau (float): Soft update coefficient for updating target networks (default: 0.005)
            initial_random_steps (int): Number of initial steps to take with random actions (default: 10000)
            updates_per_step (int): Number of updates to perform per step (default: 1)
            target_update_interval (int): Interval for updating target networks (default: 1)
            hidden_dim (int): Dimension of hidden layers in neural networks (default: 128)
            learning_rate (float): Learning rate for the optimizer (default: 0.0001)
            optimizer (str): Name of the optimizer to use, one of "adam", "rmsprop", or "sgd" (default: "adam")
        """
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space

        self.env = env
        self.memory = ReplayBuffer(memory_size, batch_size, seed)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.target_update_interval = target_update_interval
        self.initial_random_steps = initial_random_steps
        self.updates_per_step = updates_per_step
        self.seed = seed

        # Check for GPU availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        # Critic networks
        self.critic = QNetwork(obs_dim, action_dim.shape[0], hidden_dim).to(self.device)
        self.critic_target = QNetwork(obs_dim, action_dim.shape[0], hidden_dim).to(self.device)
        self.hard_update(self.critic_target, self.critic)

        # Entropy related variables
        self.target_entropy = -torch.prod(torch.tensor(action_dim.shape, dtype = torch.long, device = self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad = True, device = self.device)
        self.alpha = self.log_alpha.exp()

        # Policy network
        self.policy = GaussianPolicy(obs_dim, action_dim.shape[0], hidden_dim, action_dim).to(self.device)

        # Optimizer setup
        optimizer.lower()
        if optimizer == "adam":
            self.critic_optim = optim.Adam(self.critic.parameters(), lr = learning_rate)
            self.alpha_optim = optim.Adam([self.log_alpha], lr = learning_rate)
            self.policy_optim = optim.Adam(self.policy.parameters(), lr = learning_rate)
        if optimizer == "rmsprop":
            self.critic_optim = optim.RMSprop(self.critic.parameters(), lr = learning_rate)
            self.alpha_optim = optim.RMSprop([self.log_alpha], lr = learning_rate)
            self.policy_optim = optim.RMSprop(self.policy.parameters(), lr = learning_rate)
        if optimizer == "sgd":
            self.critic_optim = optim.SGD(self.critic.parameters(), lr = learning_rate)
            self.alpha_optim = optim.SGD([self.log_alpha], lr = learning_rate)
            self.policy_optim = optim.SGD(self.policy.parameters(), lr = learning_rate)
        
        # Total steps count
        self.total_step = 0

        # Transition to store in memory
        self.transition = list()

        # Mode: train / test
        self.is_test = False
    
    def select_action(self, state, evaluate = False):
        """
        Select an action from the input state.
        """
        # If initial random action should be conducted
        if self.total_step < self.initial_random_steps and not self.is_test:
            action = self.env.action_space.sample()
        else:
            state_t = torch.tensor(np.array(state), dtype = torch.float, device = self.device).unsqueeze(0)
            if evaluate == False:
                action, _, _ = self.policy.sample(state_t)
            else:
                _, _, action = self.policy.sample(state_t)
            action = action.detach().cpu().numpy()[0]
        
        self.transition = [state, action]

        return action
    
    def step(self, action):
        """
        Take an action and return the response of the env.
        """
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        
        if not self.is_test:
            mask = 1 if self.total_step == self.env._max_episode_steps else float(not done)
            self.transition += [reward, next_state, mask]
            self.memory.push(*self.transition)

        return next_state, reward, done
    
    def update_model(self, updates):
        """
        Update the SAC agent's critic and policy networks.
        """
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = self.memory.sample()

        state_batch = torch.tensor(state_batch, dtype = torch.float, device = self.device).to(self.device)
        next_state_batch = torch.tensor(next_state_batch, dtype = torch.float, device = self.device).to(self.device)
        action_batch = torch.tensor(action_batch, dtype = torch.float, device = self.device).to(self.device)
        reward_batch = torch.tensor(reward_batch, dtype = torch.float, device = self.device).to(self.device).unsqueeze(1)
        mask_batch = torch.tensor(mask_batch, dtype = torch.float, device = self.device).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)

        # Compute critic losses
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        # Update critic networks
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        # Sample actions and compute policy loss
        pi, log_pi, _ = self.policy.sample(state_batch)
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        # Update policy network
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # Compute alpha loss
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        # Update alpha parameter
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        # Update alpha value and store for logging
        self.alpha = self.log_alpha.exp()
        alpha_tlogs = self.alpha.clone() # For TensorboardX logs

        # Soft update target networks
        if updates % self.target_update_interval == 0:
            self.soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()
    
    def train(self, num_frames, plotting_interval = 1000):
        """
        Train the agent.
        """
        self.is_test = False
        
        state, _ = self.env.reset(seed = self.seed)
        critic_1_loss, critic_2_loss, policy_loss, ent_loss, alphas = [], [], [], [], []
        scores = []
        score = 0
        updates = 0
        
        for self.total_step in range(1, num_frames + 1):
            # At the halfway point, reduce learning rate by a tenth
            if self.total_step == int(num_frames // 2):
                adjust_learning_rate(self.critic_optim, 0.1)
                adjust_learning_rate(self.alpha_optim, 0.1)
                adjust_learning_rate(self.policy_optim, 0.1)

            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

            # If episode ends
            if done:
                state, _ = self.env.reset(seed = self.seed)
                scores.append(score)
                score = 0

            # If training is ready
            if (len(self.memory) >= self.batch_size and self.total_step > self.initial_random_steps):
                for i in range(self.updates_per_step):
                    losses = self.update_model(updates)
                    critic_1_loss.append(losses[0])
                    critic_2_loss.append(losses[1])
                    policy_loss.append(losses[2])
                    ent_loss.append(losses[3])
                    alphas.append(losses[4])
                    updates += 1
            
            # Plotting and saving results
            if self.total_step % plotting_interval == 0:
                self._plot(scores, critic_1_loss, critic_2_loss, policy_loss, ent_loss, alphas)
                self._save(scores, critic_1_loss, critic_2_loss, policy_loss, ent_loss, alphas)

                print(f"Frame: {self.total_step}, Mean of last 10 rewards: {np.mean(scores[-10:])}")

                # Create checkpoint folder
                if not os.path.exists("checkpoints"):
                    os.makedirs("checkpoints")

                # Save the model checkpoint
                checkpoint_name = f"checkpoint_sac_latest.pth.tar"
                checkpoint_path = os.path.join("checkpoints", checkpoint_name)
                torch.save({"policy_state_dict": self.policy.state_dict(), "critic_state_dict": self.critic.state_dict(), "critic_target_state_dict": self.critic_target.state_dict(), "critic_optimizer_state_dict": self.critic_optim.state_dict(), "policy_optimizer_state_dict": self.policy_optim.state_dict(), "rewards": scores, "frame": self.total_step}, checkpoint_path)

        print("Training successfully completed.")

        self.env.close()
        
    def test(self, video_folder = "sac_agent_video"):
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
            action = self.select_action(state, evaluate = True)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward
        
        print("Score: ", score)
        self.env.close()
        
        # Reset
        self.env = naive_env
    
    def soft_update(self, target, source, tau):
        """
        Perform a soft update of the target network parameters towards the source network parameters.
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def hard_update(self, target, source):
        """
        Perform a hard update of the target network parameters to be identical to the source network parameters.
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    
    def _save(self, rewards, critic_1_loss, critic_2_loss, policy_loss, ent_loss, alphas):
        """
        Save training results to a pickle file.
        """
        # Save results to a file
        with open("sac-results.pkl", "wb") as f:
            pickle.dump(rewards, f)
            pickle.dump(critic_1_loss, f)
            pickle.dump(critic_2_loss, f)
            pickle.dump(policy_loss, f)
            pickle.dump(ent_loss, f)
            pickle.dump(alphas, f)
    
    def _plot(self, rewards, critic_1_loss, critic_2_loss, policy_loss, ent_loss, alphas, moving_average_window = 100):
        """
        Plot training curves.
        """
        plt.figure(figsize = (40, 6))

        # Combined plot of rewards, moving average, policy, critic, entropy losses, and alpha
        plt.subplot(161)
        plt.title("SAC Rewards Per Episode")
        plt.plot(rewards, label = "Reward")
        if len(rewards) >= moving_average_window:
            plt.plot(moving_average(rewards, moving_average_window), label = "Moving Average", color = "red")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        
        plt.subplot(162)
        plt.title("Policy Loss")
        plt.plot(policy_loss, label = "Policy Loss")
        plt.xlabel("Frame")
        plt.ylabel("Loss")

        plt.subplot(163)
        plt.title("Critic 1 Loss")
        plt.plot(critic_1_loss)
        plt.xlabel("Frame")
        plt.ylabel("Loss")

        plt.subplot(164)
        plt.title("Critic 2 Loss")
        plt.plot(critic_2_loss)
        plt.xlabel("Frame")
        plt.ylabel("Loss")

        plt.subplot(165)
        plt.title("Entropy Loss")
        plt.plot(ent_loss)
        plt.xlabel("Frame")
        plt.ylabel("Loss")

        plt.subplot(166)
        plt.title("Alpha")
        plt.plot(alphas)
        plt.xlabel("Frame")
        plt.ylabel("Alpha")

        plt.savefig("sac_plot.pdf")
        plt.close()

        # Individual plot of rewards and moving average
        plt.figure(figsize = (20, 6))
        plt.title("SAC Rewards Per Episode")
        plt.plot(rewards, label = "Reward")
        if len(rewards) >= moving_average_window:
            plt.plot(moving_average(rewards, moving_average_window), label = "Moving Average", color = "red")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        plt.savefig("sac_plot_reward.pdf")
        plt.close()

    def _load_checkpoint(self, checkpoint_path, include_optimiser = True):
        """
        Load the models from a checkpoint.
        """
        model = torch.load(checkpoint_path, map_location = self.device)
        self.policy.load_state_dict(model["policy_state_dict"])
        self.critic.load_state_dict(model["critic_state_dict"])
        self.critic_target.load_state_dict(model["critic_target_state_dict"])
        
        if include_optimiser == True:
            self.policy_optim.load_state_dict(model["policy_optimizer_state_dict"])
            self.critic_optim.load_state_dict(model["critic_optimizer_state_dict"])
