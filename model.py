import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor network"""
    
    def __init__(self, state_size,
                 action_size, fc1_size=400,
                 fc2_size=300, seed=1):
        """Initialize paramteres and build model
        Params
        ======
            state_size (int): Dimension of state
            action_size (int): Dimension of action
            fc1_size (int): size of 1st hidden layer
            fc2_size (int): size of 2nd hidden layer
            seed (int): random seed
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.fc1 = nn.Linear(state_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.output = nn.Linear(fc2_size, action_size)
        self.reset_parameters()
        
    def forward(self, state):
        """Build a network which maps states to deterministic continuous actions"""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))       
        return F.tanh(self.output(x))
    
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.output.weight.data.uniform_(-3e-3, 3e-3)
        

class Critic(nn.Module):
    """Critic network"""
    
    def __init__(self, state_size,
                 action_size, fc1_state_size=400,
                 fc2_size = 300, seed=1):
        """Initialize paramteres and build model
        Params
        ======
            state_size (int): Dimension of state
            action_size (int): Dimension of action
            fc1_state_size (int): size of 1st hidden_layer 
                (actions are not included in the first layer)
            fc2_size (int): size of 2nd hidden layer
            seed (int): random seed
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.fc1 = nn.Linear(state_size, fc1_state_size)
        self.fc2 = nn.Linear(fc1_state_size + action_size, fc2_size) #include actions
        self.output = nn.Linear(fc2_size, 1)
        self.reset_parameters()
        
    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = F.relu(self.fc1(state))
        x = torch.cat((x, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.output(x)

    
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.output.weight.data.uniform_(-3e-3, 3e-3)