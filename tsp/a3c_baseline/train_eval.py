from tsp_env import TSPEnv
from model import ActorCriticModel
from agent import A3C_Agent

import numpy as np
import threading

# Define the distance matrix for the cities
#cities = np.array(list(range()))
distances = np.array([
    [0, 2, 2, 5, 9, 3],
    [2, 0, 4, 6, 7, 8],
    [2, 4, 0, 8, 6, 3],
    [5, 6, 8, 0, 4, 9],
    [9, 7, 6, 4, 0, 10],
    [3, 8, 3, 9, 10, 0]
])  # Your distance matrix

env = TSPEnv(distances)
#num_actions = len(cities)
num_actions = distances.shape[0]
state_size = num_actions + 1  # State includes current city and visited cities

global_model = ActorCriticModel(num_actions)
global_model.build(input_shape=(None, state_size))

num_workers = 4
threads = []

for _ in range(num_workers):
    worker = A3C_Agent(global_model, num_actions)
    thread = threading.Thread(target=worker.train, args=(env,))
    thread.start()
    threads.append(thread)

for thread in threads:
    thread.join()