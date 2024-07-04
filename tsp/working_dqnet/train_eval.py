import numpy as np
from tsp_env import TSPEnv
from agent import DQN_Agent
import threading

# Example usage
distance_matrix = np.array([
    [0, 2, 9, 10],
    [1, 0, 6, 4],
    [15, 7, 0, 8],
    [6, 3, 12, 0]
])

num_actions = distance_matrix.shape[0]
state_size = num_actions + 1 

env = TSPEnv(distance_matrix, start_city=0, end_city=3)

num_workers = 1
threads = []
results = []

def run_worker(agent, env):
    best_path, best_reward = agent.train(env)
    results.append((best_path, best_reward))

for _ in range(num_workers):
    worker = DQN_Agent(state_size, num_actions)
    thread = threading.Thread(target=run_worker, args=(worker, env))
    thread.start()
    threads.append(thread)

for thread in threads:
    thread.join()

# After training, find the best result among all workers
best_path, best_reward = max(results, key=lambda x: x[1])
print(f"Best Path: {best_path}")
print(f"Best Reward: {best_reward}")