import numpy as np
import gym
from gym import spaces

class TSPEnv(gym.Env):
    def __init__(self, cities):
        super(TSPEnv, self).__init__()
        self.cities = cities
        self.num_cities = len(cities)
        self.action_space = spaces.Discrete(self.num_cities)
        self.observation_space = spaces.Box(low=0, high=self.num_cities, shape=(self.num_cities,), dtype=np.int32)
        self.state = None
        self.min_distance, self.max_distance = self.calculate_min_max_distance()
        self.reset()

    def reset(self):
        self.state = np.random.permutation(self.num_cities)
        return self.state

    def step(self, action):
        current_index = np.where(self.state == action)[0][0]
        next_index = (current_index + 1) % self.num_cities
        self.state[current_index], self.state[next_index] = self.state[next_index], self.state[current_index]
        total_distance = self.calculate_total_distance(self.state)
        reward = self.calculate_reward(total_distance)
        done = False  # You can define a condition to end the episode
        self.render()
        return self.state, reward, done, {}

    def calculate_total_distance(self, state):
        total_distance = 0
        for i in range(self.num_cities):
            city_a = self.cities[state[i]]
            city_b = self.cities[state[(i + 1) % self.num_cities]]
            total_distance += np.linalg.norm(np.array(city_a) - np.array(city_b))
        return total_distance

    def calculate_reward(self, distance):
        normalized_distance = (distance - self.min_distance) / (self.max_distance - self.min_distance)
        return 100 * (1 - normalized_distance)  # Reward in the range 0 to 100

    def calculate_min_max_distance(self):
        all_distances = []
        for perm in np.random.permutation([np.arange(self.num_cities) for _ in range(1000)]):
            all_distances.append(self.calculate_total_distance(perm))
        return min(all_distances), max(all_distances)

    def render(self, mode='human'):
        print("Current state:", self.state)
        print("Current route:", [self.cities[i] for i in self.state])
        print("Total distance:", self.calculate_total_distance(self.state))
