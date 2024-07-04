import numpy as np
import matplotlib.pyplot as plt

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
        self.route = [self.current_city]  # Track the route taken
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
            self.route.append(action)  # Update the route
            done = all(self.get_state())

        return self.get_state(), reward, done, {}

    def render(self, mode='human'):
        print("Current state:", self.get_state())
        print("Visited cities:", self.visited)
        print("Current route:", self.route)
        print("Total distance:", self.total_distance)
        
        if mode == 'human':
            self._plot_route()

    def _plot_route(self):
        # Plotting the route
        route_cities = [self.cities[i] for i in self.route]
        x, y = zip(*route_cities)

        plt.figure(figsize=(8, 6))
        plt.plot(x, y, 'o-', markersize=10, label='Route')
        for i, city in enumerate(self.cities):
            plt.text(city[0], city[1], f'{i}', fontsize=12, ha='right')
        plt.plot(self.cities[0][0], self.cities[0][1], 'go', markersize=12, label='Start (City 0)')
        plt.title('Current TSP Route')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.legend()
        plt.grid(True)
        plt.show()