import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import threading

class TSPEnv:
    def __init__(self, distances):
        self.distances = distances
        self.num_cities = len(distances)
        self.max_distance = np.max(distances)
        self.min_distance = np.min(distances)
        self.reset()

    def reset(self):
        self.visited = [False] * self.num_cities
        self.current_city = np.random.randint(self.num_cities)
        self.visited[self.current_city] = True
        self.path = [self.current_city]
        return self.get_state()

    def get_state(self):
        return (self.current_city, tuple(self.visited))

    def step(self, action):
        if self.visited[action]:
            return self.get_state(), 0, True, {}  # Small penalty for revisiting a city

        distance = self.distances[self.current_city][action]
        normalized_distance = (distance - self.min_distance) / (self.max_distance - self.min_distance) * 100
        reward = 100 - normalized_distance  # Normalize reward to be between 0 and 100

        self.visited[action] = True
        self.current_city = action
        self.path.append(action)

        done = all(self.visited)
        if done:
            return_to_start_distance = self.distances[self.current_city][self.path[0]]
            normalized_return_to_start_distance = (return_to_start_distance - self.min_distance) / (self.max_distance - self.min_distance) * 100
            reward += 100 - normalized_return_to_start_distance  # Reward for completing the tour
        
        return self.get_state(), reward, done, {}
