import numpy as np

class TSPEnv:
    def __init__(self, cities):
        self.cities = cities
        self.num_cities = len(cities)
        self.distance_matrix = self.compute_distance_matrix(cities)
        self.reset()

    def compute_distance_matrix(self, cities):
        num_cities = len(cities)
        distance_matrix = np.zeros((num_cities, num_cities))
        for i in range(num_cities):
            for j in range(num_cities):
                distance_matrix[i][j] = np.linalg.norm(np.array(cities[i]) - np.array(cities[j]))
        return distance_matrix

    def reset(self):
        self.current_city = 0
        self.visited = [False] * self.num_cities
        self.visited[self.current_city] = True
        self.total_distance = 0
        self.num_visited = 1
        return self.get_state()

    def get_state(self):
        return [self.current_city] + self.visited

    def step(self, action):
        if self.visited[action]:
            # Penalty for revisiting a city
            reward = -100
            done = True
        else:
            # Calculate reward as negative distance to incentivize shorter paths
            reward = -self.distance_matrix[self.current_city][action]
            self.total_distance += -reward  # Since reward is negative distance
            self.current_city = action
            self.visited[action] = True
            self.num_visited += 1
            done = self.num_visited == self.num_cities

        return self.get_state(), reward, done, {}

    def render(self, mode='human'):
        print("Current state:", self.state)
        print("Current route:", [self.cities[i] for i in self.state])
        print("Total distance:", self.calculate_total_distance(self.state))

