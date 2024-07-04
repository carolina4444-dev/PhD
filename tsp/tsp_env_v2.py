import numpy as np
import gym
from gym import spaces

class TSPEnv(gym.Env):
    def __init__(self, distances):
        super(TSPEnv, self).__init__()
        self.distances = distances
        self.n = len(distances)
        self.action_space = spaces.Discrete(self.n)
        self.observation_space = spaces.Box(low=-1, high=self.n-1, shape=(self.n,), dtype=np.int32)
        self.reset()

    def reset(self):
        self.visited = [False] * self.n
        self.visited[0] = True  # Start from the first city
        self.current_city = 0
        self.route = [0] + [-1] * self.n  # Initial state with placeholders, +1 to accommodate return to start
        self.total_distance = 0
        self.steps = 0
        return np.array(self.route)

    # def step(self, action):
    #     if self.visited[action] or (action == 0 and not all(self.visited[1:])):
    #         reward = 0  # Penalize for revisiting a city or returning to the start before visiting all cities
    #         done = False
    #     else:
    #         distance = self.distances[self.current_city][action]
    #         normalized_distance = distance / np.max(self.distances)
    #         reward = 100 - normalized_distance * 100  # Normalize the reward
    #         self.total_distance += distance
    #         self.visited[action] = True
    #         self.route[self.steps + 1] = action  # Update the route with the new city
    #         self.current_city = action
    #         self.steps += 1
    #         done = self.steps == self.n - 1
    #         if done:
    #             # Complete the tour by returning to the starting city
    #             distance = self.distances[self.current_city][0]
    #             normalized_distance = distance / np.max(self.distances)
    #             reward += 100 - normalized_distance * 100
    #             self.total_distance += distance
    #             self.route[self.steps + 1] = 0  # Add the start city to the end of the route
    #             done = True

    #     return np.array(self.route), reward, done, {}

    def step(self, action):
        if self.visited[action] or (action == 0 and not all(self.visited[1:])):
            reward = 0  # Penalize for revisiting a city or returning to the start before visiting all cities
            done = False
        else:
            distance = self.distances[self.current_city][action]
            self.total_distance += distance
            self.visited[action] = True
            self.route[self.steps + 1] = action  # Update the route with the new city
            self.current_city = action
            self.steps += 1
            done = self.steps == self.n - 1
            if done:
                # Complete the tour by returning to the starting city
                distance = self.distances[self.current_city][0]
                self.total_distance += distance
                self.route[self.steps + 1] = 0  # Add the start city to the end of the route
                done = True

            # Calculate reward as negative of increase in total distance, normalized to fit within 0 to 100
            max_distance = np.max(self.distances)
            normalized_distance = distance / max_distance
            reward = 100 - normalized_distance * 100

        return np.array(self.route), reward, done, {}



    def render(self, mode='human'):
        print("Route: {}, Total distance: {}".format(self.route, self.total_distance))

    def calculate_distance(self, route):
        total_distance = 0
        for i in range(len(route) - 1):
            total_distance += self.distances[route[i]][route[i + 1]]
        total_distance += self.distances[route[-1]][route[0]]
        return total_distance
