import numpy as np
import gym
from gym import spaces

class TSPEnv(gym.Env):
    def __init__(self, distances):
        super(TSPEnv, self).__init__()
        self.distances = distances
        self.n = len(distances)
        self.action_space = spaces.Discrete(self.n)
        self.observation_space = spaces.Box(low=0, high=self.n, shape=(self.n,), dtype=np.int32)
        self.reset()

    def reset(self):
        self.visited = [False] * self.n
        self.visited[0] = True  # Start from the first city
        self.current_city = 0
        self.route = [0]
        self.total_distance = 0
        return np.array(self.route)

    def step(self, action):
        if self.visited[action] or action == 0:
            reward = -1  # Penalize for revisiting a city or returning to the start before visiting all cities
            done = False
        else:
            reward = -self.distances[self.current_city][action]
            self.total_distance += self.distances[self.current_city][action]
            self.visited[action] = True
            self.route.append(action)
            self.current_city = action
            done = all(self.visited)
            if done:
                # Complete the tour by returning to the starting city
                reward -= self.distances[self.current_city][0]
                self.total_distance += self.distances[self.current_city][0]
                self.route.append(0)
                done = True

        return np.array(self.route), reward, done, {}

    def render(self, mode='human'):
        print("Route: {}, Total distance: {}".format(self.route, self.total_distance))

    def calculate_distance(self, route):
        total_distance = 0
        for i in range(len(route) - 1):
            total_distance += self.distances[route[i]][route[i + 1]]
        total_distance += self.distances[route[-1]][route[0]]
        return total_distance
