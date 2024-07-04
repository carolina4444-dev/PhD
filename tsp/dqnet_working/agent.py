import tensorflow as tf
import numpy as np
import threading
#from model import 
from model import create_q_network


class DQN_Agent:
    def __init__(self, global_model, num_actions, gamma=0.99):
        # Hyperparameters
        state_size = num_cities + 1  # Representation of state
        action_size = num_cities  # Number of possible actions
        gamma = 0.99
        epsilon = 1.0
        epsilon_min = 0.01
        epsilon_decay = 0.995
        batch_size = 64
        target_update_freq = 10
        memory_size = 10000

    def train():
        if len(memory) < batch_size:
            return
        
        minibatch = random.sample(memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        
        states = np.array(states)
        next_states = np.array(next_states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones).astype(int)
        
        target_q_values = q_network.predict(states[:, :, np.newaxis])
        next_target_q_values = target_network.predict(next_states[:, :, np.newaxis])
        
        for i in range(batch_size):
            target_q_values[i, actions[i]] = rewards[i] + (1 - dones[i]) * gamma * np.max(next_target_q_values[i])
        
        q_network.train_on_batch(states[:, :, np.newaxis], target_q_values)
