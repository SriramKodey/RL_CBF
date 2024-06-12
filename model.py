import torch
import torch.nn as nn
from torch.optim import Adam

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Critic(nn.Module):
    def __init__(self, dim_input=5, dim_output=1, dim_hidden=32, lr=0.0005):
        super(Critic, self).__init__()
        self.flatten = nn.Flatten()
        self.lossFn = nn.MSELoss()
        self.optimiser = Adam(self.parameters(), lr)
        self.network = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU,
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU,
            nn.Linear(dim_hidden, dim_output)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        #x = self.flatten(x)
        y = self.network(x)
        return y
    
class Actor(nn.Module):
    def __init__(self, dim_input=4, dim_output=1, dim_hidden=32, lr=0.0005):
        super(Actor, self).__init__()
        self.flatten = nn.Flatten()
        self.lossFn = nn.MSELoss()
        self.optimiser = Adam(self.parameters(), lr)
        self.network = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU,
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU,
            nn.Linear(dim_hidden, dim_output)
        )

    def forward(self, state):
        x = state
        #x = self.flatten(x)
        y = self.network(x)
        return y