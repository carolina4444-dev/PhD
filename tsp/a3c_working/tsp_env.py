
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import threading

class TSPEnv:
    def __init__(self, distance_matrix, start_city=0):
        self.distance_matrix = distance_matrix
        self.num_cities = len(distance_matrix)
        self.start_city = start_city
        self.reset()

    def reset(self):
        self.visited = [False] * self.num_cities
        self.current_city = self.start_city
        self.visited[self.current_city] = True
        self.steps = 0
        return (self.current_city, self.visited)

    def step(self, action):
        if self.visited[action]:
            reward = 0  # Penalty for revisiting a city
            done = True
        else:
            reward = 100 - self.distance_matrix[self.current_city, action]  # Adjust to keep within 0-100
            self.visited[action] = True
            self.current_city = action
            self.steps += 1
            done = all(self.visited) and (self.steps == self.num_cities - 1)
            if done:
                reward += 100 - self.distance_matrix[self.current_city, self.start_city]  # Return to start city
                self.visited[self.start_city] = True

        return (self.current_city, self.visited), reward, done, {}