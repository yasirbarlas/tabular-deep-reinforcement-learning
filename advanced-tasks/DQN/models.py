
import numpy as np
import random

import torch
import torch.nn as nn

# Default device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(SimpleDQN, self).__init__()
        
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
        
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        ).to(device)
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def feature_size(self):
        return self.features(torch.zeros(1, *self.input_shape, device = device)).view(1, -1).size(1)
    
    def act(self, state, epsilon):
        if random.random() > epsilon:
            if torch.is_tensor(state) == False:
                state = np.array(np.float32(state))
                state = torch.tensor(np.float32(state), dtype = torch.float, device = device).unsqueeze(0)
            q_value = self.forward(state)
            action = int(q_value.max(1)[1].data[0])
        else:
            action = random.randrange(self.num_actions)
        return action