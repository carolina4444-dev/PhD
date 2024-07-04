import numpy as np
import gym
from gym import spaces

class TSPEnv(gym.Env):
    def __init__(self, distance_matrix, start_city, end_city):
        super(TSPEnv, self).__init__()

        self.distance_matrix = distance_matrix
        self.num_cities = len(distance_matrix)
        self.start_city = start_city
        self.end_city = end_city

        self.action_space = spaces.Discrete(self.num_cities)
        self.observation_space = spaces.MultiBinary(self.num_cities)

        self.reset()

    def reset(self):
        self.visited = [False] * self.num_cities
        self.current_city = self.start_city
        self.visited[self.current_city] = True
        self.tour = [self.current_city]
        self.total_distance = 0
        return self._get_obs()

    def _get_obs(self):
        obs = np.zeros(self.num_cities)
        obs[self.tour] = 1
        return obs

    def step(self, action):
        if self.visited[action] or action == self.current_city:
            reward = -10  # Penalty for revisiting a city or staying in the same city
        else:
            self.total_distance += self.distance_matrix[self.current_city][action]
            self.current_city = action
            self.visited[action] = True
            self.tour.append(action)
            reward = -self.distance_matrix[self.current_city][action]

        done = len(self.tour) == self.num_cities
        if done:
            if self.current_city != self.end_city:
                reward -= 100  # Large penalty for not ending at the correct city
            else:
                self.total_distance += self.distance_matrix[self.current_city][self.end_city]
                reward -= self.distance_matrix[self.current_city][self.end_city]
            done = True
        return self._get_obs(), reward, done, {}

    def render(self, mode='human'):
        tour_str = ' -> '.join(map(str, self.tour))
        print(f'Tour: {tour_str} | Total Distance: {self.total_distance}')

# # Example usage
# distance_matrix = np.array([
#     [0, 2, 9, 10],
#     [1, 0, 6, 4],
#     [15, 7, 0, 8],
#     [6, 3, 12, 0]
# ])
# env = TSPEnv(distance_matrix, start_city=0, end_city=3)
# obs = env.reset()
# done = False
# while not done:
#     action = env.action_space.sample()
#     obs, reward, done, info = env.step(action)
#     env.render()
