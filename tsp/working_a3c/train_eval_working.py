import numpy as np
from tsp_env import TSPEnv
from model import get_vqvae  # Assuming this function creates your model correctly
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

# Define cities as coordinates (example)
cities = [
    [0, 0],
    [1, 1],
    [2, 0],
    [1, 2],
    [0, 1],
    [2, 1],
    [1, 0],
    [0, 2]
]

# A3C Agent
class A3CAgent:
    def __init__(self, num_cities, optimizer, gamma=0.99):
        self.num_cities = num_cities
        self.optimizer = optimizer
        self.gamma = gamma
        self.local_network = get_vqvae(num_cities + 1, num_cities)  # Including current city in input size

    def select_action(self, state, epsilon=0.0):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        policy_logits, _ = self.local_network(state)

        if np.random.rand() < epsilon:
            # Random action (exploration)
            return np.random.choice(self.num_cities)
        else:
            # Greedy action (exploitation)
            return np.random.choice(self.num_cities, p=np.squeeze(tf.nn.softmax(policy_logits).numpy()))

    def compute_loss(self, states, actions, rewards):
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)

        with tf.GradientTape() as tape:
            logits, values = self.local_network(states)
            values = tf.squeeze(values)

            # Compute discounted rewards
            discounted_rewards = []
            cumulative_reward = 0
            for reward in reversed(rewards):
                cumulative_reward = reward + self.gamma * cumulative_reward
                discounted_rewards.insert(0, cumulative_reward)

            discounted_rewards = tf.convert_to_tensor(discounted_rewards, dtype=tf.float32)

            # Compute advantages
            advantages = discounted_rewards - values

            # Normalize advantages to stabilize training
            advantages = (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)

            # Compute policy loss using Huber loss
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=actions)
            policy_loss = tf.reduce_mean(tf.keras.losses.Huber()(advantages, neg_log_prob))

            # Compute value loss using Huber loss
            value_loss = tf.reduce_mean(tf.keras.losses.Huber()(advantages, tf.zeros_like(advantages)))

            total_loss = policy_loss + 0.5 * value_loss
        
        grads = tape.gradient(total_loss, self.local_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.local_network.trainable_variables))

        return total_loss

# Environment class with reward calculation fixed
class TSPEnv:
    def __init__(self, cities):
        self.cities = cities
        self.num_cities = len(cities)
        self.distance_matrix = self.compute_distance_matrix(cities)
        self.max_distance = np.max(self.distance_matrix)
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
            reward = 0
            done = True
        else:
            # Calculate normalized reward
            distance = self.distance_matrix[self.current_city][action]
            normalized_reward = 100 * (1 - (distance / self.max_distance))
            reward = normalized_reward
            self.total_distance += distance
            self.current_city = action
            self.visited[action] = True
            self.num_visited += 1
            done = self.num_visited == self.num_cities

            reward = reward*self.num_visited

        return self.get_state(), reward, done, {}

    def render(self, mode='human'):
        print("Current city:", self.current_city)
        print("Visited cities:", self.visited)
        print("Total distance:", self.total_distance)

# Training loop
def train_a3c(env, optimizer, max_episodes):
    agent = A3CAgent(env.num_cities, optimizer)
    epsilon = 1.0  # Initial exploration parameter
    epsilon_min = 0.01  # Minimum exploration parameter
    epsilon_decay = 0.995  # Decay rate for exploration parameter
    max_steps_per_episode = env.num_cities

    best_road = None
    best_total_distance = float('inf')

    mean_reward = []

    for episode in range(max_episodes):
        state = env.reset()
        visited_cities = [False] * env.num_cities
        visited_cities[state[0]] = True  # Mark the starting city as visited
        road_tracking = [state[0]]

        states, actions, rewards = [], [], []
        
        done = False
        for step in range(max_steps_per_episode):
            action = agent.select_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            
            # Track road
            road_tracking.append(next_state[0])
            visited_cities[next_state[0]] = True

            states.append(state.copy())  # Include current city and visited cities
            actions.append(action)
            rewards.append(reward)

            state = next_state

            

            if done:
                break

        # Update the best road found so far
        if env.total_distance < best_total_distance and all(visited_cities):
            best_total_distance = env.total_distance
            best_road = road_tracking.copy()

        loss = agent.compute_loss(states, actions, rewards)
        print(f"Episode {episode + 1}/{max_episodes}, Loss: {loss.numpy()}, Total Distance: {env.total_distance}")
        print(f"Visited cities: {visited_cities}")
        print(f"Road tracking: {road_tracking}")

        # Decay epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        mean_reward.append(np.mean(rewards))
        print('total reward', mean_reward[-1])


    print(f"Best Road: {best_road}, Best Total Distance: {best_total_distance}")

    return mean_reward

# Example usage
# Define environment and global network
env = TSPEnv(cities)
optimizer = Adam(learning_rate=0.0001)

# Train A3C
mean_reward = train_a3c(env, optimizer, max_episodes=10000)
env.render()

import pickle
# Step 2: Pickle the list of arrays to a file
with open('mean_reward_a3c_tsp.pkl', 'wb') as f:
    pickle.dump(mean_reward, f)

print("List of arrays has been pickled to 'list_of_arrays.pkl'")
