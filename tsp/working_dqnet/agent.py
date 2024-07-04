import tensorflow as tf
import numpy as np
import threading
#from model import ActorCriticModel
from model import create_q_network
from collections import deque
import random



class DQN_Agent:
    def __init__(self, state_size, action_size, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 batch_size=64, target_update_freq=10, memory_size=10000):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.memory = deque(maxlen=memory_size)
        
        self.q_network = create_q_network(state_size, action_size)
        self.target_network = create_q_network(state_size, action_size)
        self.target_network.set_weights(self.q_network.get_weights())
    
    
    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        q_values = self.q_network.predict(state[np.newaxis, :, np.newaxis])
        return np.argmax(q_values[0])
    
    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        
        states = np.array(states)
        next_states = np.array(next_states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones).astype(int)
        
        target_q_values = self.q_network.predict(states[:, :, np.newaxis])
        next_target_q_values = self.target_network.predict(next_states[:, :, np.newaxis])
        
        for i in range(self.batch_size):
            target_q_values[i, actions[i]] = rewards[i] + (1 - dones[i]) * self.gamma * np.max(next_target_q_values[i])
        
        self.q_network.train_on_batch(states[:, :, np.newaxis], target_q_values)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def update_target_network(self):
        self.target_network.set_weights(self.q_network.get_weights())
    
    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

