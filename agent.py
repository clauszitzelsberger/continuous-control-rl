import numpy as np
import random
from collections import namedtuple, deque
import torch
import torch.nn.functional as F
import torch.optim as optim

from model import Actor, Critic

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu)

class Agent():
    
    def __init__(self, state_size,
                 action_size, buffer_size=int(1e5),
                 batch_size=64, gamma=.99,
                 tau=.001, lr_a=1e-4,
                 lr_c=1e-3, weight_decay=1e-2):
        
        """Initialize an Agent object
        
        Params
        =====
            state_size (int): Dimension of states
            action_size (int): Dimension of actions
            buffer_size (int): size of replay buffer
            batch_size (int): size of sample
            gamma (float): discount factor
            tau (float): (soft) update of target parameters
            lr_a (float): learning rate of actor
            lr_c (float): learning rate of critic
            weight_decay (float): L2 weight decay
            seed (int): random seed
        """
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        # Hyperparameters
        self.buffer_size = buffer_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.weight_decay = weight_decy
        
        # Actor networks
        self.actor_local = \
            Actor(state_size, action_size, seed=seed).to(device)
        self.actor_target = \
            Actor(state_size, action_size, seed=seed).to(device)
        self.actor_optimizer = \
            optim.Adam(self.actor_local.parameters(), lr=lr_a)
            
        # Critic networks
        self.critic_local = \
            Critic(state_size, action_size, seed=seed).to(device)
        self.critic_target = \
            Critic(state_size, action_size, seed=seed).to(device)
        self.critic_optimizer = \
            optim.Adam(self.critic_local.parameters(), lr=lr_c)
            
        # Replay buffer
        self.memory = ReplayBuffer(actions_size, buffer_size, batch_size, seed)
        
        # Noise process
        self.noise = OUNoise(action_size, seed)
        
    def step(self):
        pass
    
    def act(self):
        pass
    
    def __reset(self):
        pass
    
    def __learn(self):
        pass
    
    def __soft_update(self):
        

class ReplayBuffer:
    
    def __init__(self):
        pass
    
    def add(self):
        pass
    
    def sample(self):
        pass
    
    def __len__(self):
        pass
 
           
class OUNoise:
    
    def __init__(self):
        pass
    
    def reset(self):
        pass
    
    def sample(self):
        pass