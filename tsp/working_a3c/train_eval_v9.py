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
    [1, -1]
]

# Create the environment
env = TSPEnv(cities)

# A3C Agent
class A3CAgent:
    def __init__(self, num_cities, global_network, optimizer, gamma=0.99):
        self.num_cities = num_cities
        self.global_network = global_network
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
            return np.random.choice(self.num_cities, p=tf.nn.softmax(policy_logits).numpy().squeeze())

    def compute_loss(self, states, actions, rewards):
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)

        with tf.GradientTape() as tape:
            logits, values = self.local_network(states)
            values = tf.squeeze(values)

            # Compute advantages
            returns = tf.concat([rewards[1:], [0]], axis=0)  # Bootstrap from next state value
            advantages = rewards + self.gamma * returns - values

            # Normalize advantages to stabilize training
            advantages = (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)

            # Compute policy loss
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=actions)
            policy_loss = tf.reduce_mean(advantages * neg_log_prob)

            # Compute value loss
            value_loss = tf.reduce_mean(tf.square(advantages))

            total_loss = policy_loss + 0.5 * value_loss
        
        grads = tape.gradient(total_loss, self.local_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.local_network.trainable_variables))

        return total_loss

# Training loop
def train_a3c(env, global_network, optimizer, max_episodes=1000):
    agent = A3CAgent(env.num_cities, global_network, optimizer)
    epsilon = 1.0  # Initial exploration parameter
    epsilon_min = 0.01  # Minimum exploration parameter
    epsilon_decay = 0.995  # Decay rate for exploration parameter
    max_steps_per_episode = env.num_cities

    for episode in range(max_episodes):
        state = env.reset()
        states, actions, rewards = [], [], []

        for step in range(max_steps_per_episode):
            action = agent.select_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)

            states.append(state.copy())  # Include current city and visited cities
            actions.append(action)
            rewards.append(reward)

            if done:
                break

            state = next_state

        loss = agent.compute_loss(states, actions, rewards)
        print(f"Episode {episode + 1}/{max_episodes}, Loss: {loss.numpy()}")

        # Update global network periodically (optional)
        global_network.set_weights(agent.local_network.get_weights())

        # Decay epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
    
    print('last state: ', )

# Example usage
# Define environment and global network
env = TSPEnv(cities)
global_network = get_vqvae(env.num_cities + 1, env.num_cities)
optimizer = Adam(learning_rate=0.0001)

# Train A3C
train_a3c(env, global_network, optimizer, max_episodes=100)
