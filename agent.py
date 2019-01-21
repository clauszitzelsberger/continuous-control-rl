import numpy as np
import random
import copy
from collections import namedtuple, deque
import torch
import torch.nn.functional as F
import torch.optim as optim

from model import Actor, Critic

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Agent():
    
    def __init__(self, state_size,
                 action_size, n_agents=1,
                 buffer_size=int(1e6), batch_size=128, 
                 gamma=.99, tau=1e-3, 
                 lr_a=1e-4, lr_c=1e-3, 
                 weight_decay=1e-2, update_local=4, 
                 seed=1):
        
        """Initialize an Agent object
        
        Params
        =====
            state_size (int): Dimension of states
            action_size (int): Dimension of actions
            n_agents (int): Number of agents
            buffer_size (int): size of replay buffer
            batch_size (int): size of sample
            gamma (float): discount factor
            tau (float): (soft) update of target parameters
            lr_a (float): learning rate of actor
            lr_c (float): learning rate of critic
            weight_decay (float): L2 weight decay
            update_local (int): update local network after every x steps
            seed (int): random seed
        """
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.n_agents = n_agents
        
        # Hyperparameters
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.weight_decay = weight_decay
        self.update_local = update_local
        
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
            optim.Adam(self.critic_local.parameters(), lr=lr_c, 
                       weight_decay=weight_decay)
            
        # Replay buffer
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed)
        
        # Noise process
        self.noise = OUNoise(action_size, seed)
        
        # Time step
        self.t_step = 0
        
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay buffer
        for i in range(self.n_agents):
            self.memory.add(state[i], action[i], 
                            reward[i], next_state[i], 
                            done[i])
        
        # Learn every UPDATE LOCAL time steps
        self.t_step += 1
        if self.t_step % self.update_local == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                sample = self.memory.sample()
                self.__learn(sample, self.gamma)

    def act(self, state, add_noise=True):
        """Returns action given a state according to current policy

        Params
        ======
            state (array_like): current state
            add_noise (bool): handles exploration
        """
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        if add_noise:
            action += [self.noise.sample() for i in range(self.n_agents)]
        return np.clip(action, -1, 1)

    
    def reset(self):
        self.noise.reset()
    
    def __learn(self, sample, gamma):
        """
        Params
        ======
            sample (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = sample
        
        #----------------- Critic
        # Next actions and actions values
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local Critic network
        Q_expected = self.critic_local(states, actions)
        
        # Compute loss
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()
        
        #----------------- Actor
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        
        # Minimize loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        #----------------- update target networks
        self.__soft_update(self.critic_local, self.critic_target, self.tau)
        self.__soft_update(self.actor_local, self.actor_target, self.tau)
        
    
    def __soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param \
            in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.\
                copy_(tau*local_param.data + (1.0 - tau)*target_param.data)
             
        
        

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples in"""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batchparamteres
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = \
            namedtuple('Experience',
                       field_names=['state', 'action', 'reward', 'next_state', 'done'])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory"""
        self.memory.\
            append(self.experience(state, action, reward, next_state, done))

    def sample(self):
        """Randomly sample a batch of experiences from memory"""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.\
            from_numpy(np.vstack([e.state for e in experiences if e is not None])).\
            float().to(device)
        actions = torch.\
            from_numpy(np.vstack([e.action for e in experiences if e is not None])).\
            float().to(device)
        rewards = torch.\
            from_numpy(np.vstack([e.reward for e in experiences if e is not None])).\
            float().to(device)
        next_states = torch.\
            from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).\
            float().to(device)
        dones = torch.\
            from_numpy(np.vstack([e.done for e in experiences if e is not None]).\
            astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)


    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
 
           
class OUNoise:
    """Ornstein-Uhlenbeck process"""
    
    def __init__(self, size, seed, mu=.0, theta=.15, sigma=.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()
    
    def reset(self):
        """Reset the internal state (=noise) to mean (mu)"""
        self.state = copy.copy(self.mu)
        
    def sample(self):
        """Update internal state and return as a noise sample"""
        x = self.state
        dx = self.theta * (self.mu - x) + \
            self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state