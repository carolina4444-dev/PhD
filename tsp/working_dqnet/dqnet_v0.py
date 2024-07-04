import numpy as np
from tsp_env import TSPEnv
import tensorflow as tf
from agent import DQN_Agent

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

# Initialize agent parameters
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
gamma = 0.99

# Epsilon-greedy parameters
epsilon = 1.0  # Initial epsilon value
epsilon_min = 0.01  # Minimum epsilon value
epsilon_decay = 0.995  # Epsilon decay rate per episode

# Initialize agent model
agent = DQN_Agent(num_actions + 1, num_actions)

# Training parameters
num_episodes = 1000
best_reward = -float('inf')
best_path = None



for episode in range(num_episodes):  # Number of episodes
    state = env.reset()  # Reset environment for each episode
    current_path = [env.start_city]
    episode_reward = 0
    done = False
    for t in range(100):  # Number of steps per episode
        current_city, visited_cities = state
        state_representation = np.zeros(num_actions + 1)
        state_representation[current_city] = 1
        for city in visited_cities[:-1]:  # Exclude the last city from negative representation
            state_representation[city] = -1
        action = agent.select_action(state_representation)
        next_state, reward, done = env.step(state, action)
        
        next_city, next_visited_cities = next_state
        next_state_representation = np.zeros(num_actions + 1)
        next_state_representation[next_city] = 1
        for city in next_visited_cities[:-1]:  # Exclude the last city from negative representation
            next_state_representation[city] = -1
        
        agent.remember(state_representation, action, reward, next_state_representation, done)
        state = next_state
        agent.train()
        if done:
            print(f"Episode {episode+1}/{1000}, Path Reward: {reward}")
            print("Path:", visited_cities[:-1])  # Print path excluding the last city
            break
    agent.decay_epsilon()
    if episode % agent.target_update_freq == 0:
        agent.update_target_network()

