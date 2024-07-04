import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import threading

class TSPEnv:
    def __init__(self, distances):
        self.distances = distances
        self.num_cities = len(distances)
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
            return self.get_state(), -100, True, {}
        
        distance = self.distances[self.current_city][action]
        self.visited[action] = True
        self.current_city = action
        self.path.append(action)
        
        reward = -distance

        done = all(self.visited)
        if done:
            reward += 1000 - self.distances[self.current_city][self.path[0]]
        
        return self.get_state(), reward, done, {}
