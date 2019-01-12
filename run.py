import torch
from unityagents import UnityEnvironment
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
from collections import deque
#import pdb

from agent import Agent
from model import QNetwork

def initialize_env(unity_file):
    # Initialize the environment
    env = UnityEnvironment(file_name=unity_file)

    # Get default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # Get state and action spaces
    env_info = env.reset(train_mode=True)[brain_name]
    state_size = len(env_info.vector_observations[0])
    action_size = brain.vector_action_space_size
    
    print('State size:', state_size)
    print('Action size:', action_size)
    
    return env, brain_name, state_size, action_size





def ddpg():
    pass

def apply():
    model = load_checkpoints(filepath)
    env_info = env.reset(train_mode=False)[brain_name]
    state = env_info.vector_observations[0]
    score = 0
    while True:
        state = torch.from_numpy(state).float().unsqueeze(0).to('cpu')
        model.eval()
        with torch.no_grad():
            action_values = model(state)
        action = np.argmax(action_values.cpu().data.numpy())
        env_info = env.step(action)[brain_name]
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        score += reward
        state = next_state
        if done:
            break
    print('Score: {}'.format(score))
    
def plot_scores(scores_dict):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for key, scores in scores_dict.items():
        scores_smoothed = gaussian_filter1d(scores, sigma=5)
        plt.plot(np.arange(len(scores)), scores_smoothed, label=key)
    plt.ylabel('smoothed Score')
    plt.xlabel('Episode #')
    plt.legend()
    plt.show()

def load_checkpoints(filepath):
    checkpoint = torch.load(filepath)
    model = QNetwork(checkpoint['state_size'],
                     checkpoint['action_size'],
                     checkpoint['hidden_layers'])
    model.load_state_dict(checkpoint['state_dict'])
    return model


if __name__ == '__main__':
    env, brain_name, state_size, action_size = \
        initialize_env('Reacher/Reacher.x86')

    env.close()