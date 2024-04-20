
import numpy as np
import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Default device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init = 0.5):
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features, device = device, dtype = torch.float, requires_grad = True))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features, device = device, dtype = torch.float, requires_grad = True))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features, device = device, dtype = torch.float))

        self.bias_mu = nn.Parameter(torch.empty(out_features, device = device, dtype = torch.float, requires_grad = True))
        self.bias_sigma = nn.Parameter(torch.empty(out_features, device = device, dtype = torch.float, requires_grad = True))
        self.register_buffer("bias_epsilon", torch.empty(out_features, device = device, dtype = torch.float))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        
        if self.bias_mu is not None:
            self.bias_mu.data.uniform_(-mu_range, mu_range)
            self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        
        if self.bias_mu is not None:
            self.bias_epsilon.copy_(epsilon_out)

    def scale_noise(self, size):
        if isinstance(size, int):
            size = (size,)
        x = torch.randn(size, device = self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

class NoisyDQN(nn.Module):
    def __init__(self, input_shape, num_actions, noisy_std = 0.5):
        super(NoisyDQN, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions
        
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size = 8, stride = 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size = 4, stride = 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size = 3, stride = 1),
            nn.ReLU()
        ).to(device)
        
        self.noisy1 = NoisyLinear(self.feature_size(), 512, std_init = noisy_std)
        self.noisy2 = NoisyLinear(512, num_actions, std_init = noisy_std)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        x = x / 255.0
        x = self.features(x)
        x = x.view(batch_size, -1)
        
        x = F.relu(self.noisy1(x))
        x = self.noisy2(x)
        return x
    
    def reset_noise(self):
        self.noisy1.reset_noise()
        self.noisy2.reset_noise()
    
    def feature_size(self):
        return self.features(torch.zeros(1, *self.input_shape, device = device)).view(1, -1).size(1)
    
    def act(self, state):
        if torch.is_tensor(state) == False:
            state = np.array(np.float32(state))
            state = torch.tensor(np.float32(state), dtype = torch.float, device = device).unsqueeze(0)
        q_value = self.forward(state)
        action = int(q_value.max(1)[1].data[0])
        return action