
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import threading

class TSPEnv:
    def __init__(self, distance_matrix):
        self.distance_matrix = distance_matrix
        self.num_cities = len(distance_matrix)
        self.reset()

    def reset(self):
        self.visited = [False] * self.num_cities
        self.current_city = np.random.randint(self.num_cities)
        self.visited[self.current_city] = True
        self.steps = 0
        return (self.current_city, self.visited)

    def step(self, action):
        if self.visited[action]:
            reward = -10  # Penalty for revisiting a city
            done = True
        else:
            reward = -self.distance_matrix[self.current_city, action]
            self.visited[action] = True
            self.current_city = action
            self.steps += 1
            done = all(self.visited) and self.steps == self.num_cities
            if done:
                reward -= self.distance_matrix[self.current_city, 0]  # Return to start city

        return (self.current_city, self.visited), reward, done, {}