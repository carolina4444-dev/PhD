import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, initializers

class DQNAgent:
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
        
        self.q_network = self.create_q_network()
        self.target_network = self.create_q_network()
        self.target_network.set_weights(self.q_network.get_weights())
    
    def create_q_network(self):
        inputs = layers.Input(shape=(self.state_size, 1))
    
        # Inception module with parallel Conv1D layers
        tower_1 = layers.Conv1D(64, 1, padding='same', activation='relu', 
                                kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05))(inputs)
        
        tower_2 = layers.Conv1D(64, 1, padding='same', activation='relu', 
                                kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05))(inputs)
        tower_2 = layers.Conv1D(64, 3, padding='same', activation='relu', 
                                kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05))(tower_2)
        
        tower_3 = layers.Conv1D(64, 1, padding='same', activation='relu', 
                                kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05))(inputs)
        tower_3 = layers.Conv1D(64, 5, padding='same', activation='relu', 
                                kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05))(tower_3)
        
        tower_4 = layers.MaxPooling1D(3, strides=1, padding='same')(inputs)
        tower_4 = layers.Conv1D(64, 1, padding='same', activation='relu', 
                                kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05))(tower_4)
        
        # Concatenate the outputs of the inception module
        concatenated = layers.Concatenate(axis=-1)([tower_1, tower_2, tower_3, tower_4])
        
        # Additional Conv1D layers
        conv = layers.Conv1D(256, 3, padding='same', activation='relu', 
                             kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05))(concatenated)
        pool = layers.MaxPooling1D(pool_size=2)(conv)
        
        # Flatten and fully connected layers
        flat = layers.Flatten()(pool)
        dense1 = layers.Dense(128, activation='relu', 
                              kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05))(flat)
        dropout1 = layers.Dropout(0.5)(dense1)
        
        # Output layer
        outputs = layers.Dense(self.action_size, 
                               kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05))(dropout1)

        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mse')
        return model
    
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

class TravelingSalesmanEnv:
    def __init__(self, distances):
        self.distances = distances
        self.num_cities = distances.shape[0]
    
    def initialize_state(self, start_city=None, end_city=None):
        if start_city is None:
            start_city = np.random.choice(self.num_cities)
        if end_city is None:
            end_city = np.random.choice(self.num_cities)
            while end_city == start_city:
                end_city = np.random.choice(self.num_cities)
        
        visited_cities = [start_city, end_city]
        state = (end_city, visited_cities)  # Start with end city to enforce continuity
        return state
    
    def step(self, state, action):
        current_city, visited_cities = state
        next_city = action
        
        if next_city in visited_cities:
            # Large negative reward if the city has already been visited
            reward = -100
        else:
            # Scale the negative distance to range [0, 100]
            reward = 100 - self.distances[current_city, next_city]
        
        visited_cities.insert(-1, next_city)  # Insert next city before the last (end) city
        
        # Check if all cities have been visited except the first and last
        done = len(visited_cities) == self.num_cities
        
        return (visited_cities[-2], visited_cities), reward, done

# Main Training Loop
if __name__ == "__main__":
    distances = np.array([
        [0, 2, 2, 5, 9, 3],
        [2, 0, 4, 6, 7, 8],
        [2, 4, 0, 8, 6, 3],
        [5, 6, 8, 0, 4, 9],
        [9, 7, 6, 4, 0, 10],
        [3, 8, 3, 9, 10, 0]
    ])
    num_cities = distances.shape[0]
    
    state_size = num_cities + 1  # Representation of state
    action_size = num_cities  # Number of possible actions
    
    agent = DQNAgent(state_size, action_size)
    env = TravelingSalesmanEnv(distances)
    
    start_city = None
    end_city = None
    for episode in range(1000):  # Number of episodes
        state = env.initialize_state(start_city, end_city)
        end_city = state[0]  # Update end city for next episode
        for t in range(100):  # Number of steps per episode
            current_city, visited_cities = state
            state_representation = np.zeros(num_cities + 1)
            state_representation[current_city] = 1
            for city in visited_cities[:-1]:  # Exclude the last city from negative representation
                state_representation[city] = -1
            action = agent.select_action(state_representation)
            next_state, reward, done = env.step(state, action)
            
            next_city, next_visited_cities = next_state
            next_state_representation = np.zeros(num_cities + 1)
            next_state_representation[next_city] = 1
            for city in next_visited_cities[:-1]:  # Exclude the last city from negative representation
                next_state_representation[city] = -1
            
            agent.remember(state_representation, action, reward, next_state_representation, done)
            state = next_state
            agent.train()
            if done:
                print(f"Episode {episode+1}/{1000}, Path Reward: {reward}")
                print("Path:", visited_cities[:-1])  # Print path excluding the last city
                break
        agent.decay_epsilon()
        if episode % agent.target_update_freq == 0:
            agent.update_target_network()
