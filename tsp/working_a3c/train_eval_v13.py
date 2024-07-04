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
    [1, 2]
]

# A3C Agent
class A3CAgent:
    def __init__(self, num_cities, optimizer, gamma=0.99):
        self.num_cities = num_cities
        self.optimizer = optimizer
        self.gamma = gamma
        self.local_network = get_vqvae(num_cities + 1, num_cities)  # Including current city in input size
        self.best_route = None
        self.best_distance = float('inf')

    def select_action(self, state, epsilon=0.0):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        policy_logits, _ = self.local_network(state)

        if np.random.rand() < epsilon:
            # Random action (exploration)
            return np.random.choice(self.num_cities)
        else:
            # Greedy action (exploitation)
            return np.random.choice(self.num_cities, p=tf.nn.softmax(policy_logits).numpy().squeeze())

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

    def update_best_route(self, route, total_distance):
        if total_distance < self.best_distance:
            self.best_route = route
            self.best_distance = total_distance

# Training loop
def train_a3c(env, optimizer, max_episodes):
    agent = A3CAgent(env.num_cities, optimizer)
    epsilon = 1.0  # Initial exploration parameter
    epsilon_min = 0.01  # Minimum exploration parameter
    epsilon_decay = 0.995  # Decay rate for exploration parameter
    max_steps_per_episode = env.num_cities + 1

    for episode in range(max_episodes):
        state = env.reset()
        visited_cities = [False] * env.num_cities
        visited_cities[state[0]] = True  # Mark the starting city as visited
        road_tracking = [state[0]]

        states, actions, rewards = [], [], []

        total_distance = 0
        done = False

        #for step in range(max_steps_per_episode):
        while not done:
            action = agent.select_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            visited_cities[next_state[0]] = True
            road_tracking.append(next_state[0])

            states.append(state.copy())  # Include current city and visited cities
            actions.append(action)
            rewards.append(reward)
            total_distance += reward  # Since reward is negative distance

            if done:
                break

            state = next_state

        loss = agent.compute_loss(states, actions, rewards)
        # print(f"Episode {episode + 1}/{max_episodes}, Loss: {loss.numpy()}")
        # print(f"Visited cities: {visited_cities}")
        # print(f"Road tracking: {road_tracking}")
        print('reward', reward)

        # Update best route if this episode's route is better
        agent.update_best_route(road_tracking, total_distance)

        # Decay epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

    print("Best route found:", agent.best_route)
    print("Best total distance:", agent.best_distance)

# Example usage
# Define environment and global network
env = TSPEnv(cities)
optimizer = Adam(learning_rate=0.0001)

# Train A3C
train_a3c(env, optimizer, max_episodes=1000)
env.render()
