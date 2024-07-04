import numpy as np
from tsp_env import TSPEnv
from model import get_vqvae
from agent import A3C_Agent
import threading

# Example usage
# Define cities as coordinates (example)
cities = [
    [0, 0],
    [1, 1],
    [2, 0],
    [1, -1]
]

# Create the environment
env = TSPEnv(cities)

# Calculate the number of actions as the number of cities
num_actions = len(cities)

# State size includes the current city and the visited cities
state_size = num_actions + 1

# Initialize the global model with the correct input shape
global_model = get_vqvae(state_size, num_actions)
global_model.build(input_shape=(None, state_size))

# Number of workers (threads) to run in parallel
num_workers = 1
threads = []
results = []

def run_worker(agent, env):
    best_path, best_reward = agent.train(env)
    results.append((best_path, best_reward))

# Create and start threads
for _ in range(num_workers):
    worker = A3C_Agent(global_model, num_actions, env)
    thread = threading.Thread(target=run_worker, args=(worker, env))
    thread.start()
    threads.append(thread)

# Join threads to ensure all threads have completed
for thread in threads:
    thread.join()

# After training, find the best result among all workers
best_path, best_reward = max(results, key=lambda x: x[1])
print(f"Best Path: {best_path}")
print(f"Best Reward: {best_reward}")
