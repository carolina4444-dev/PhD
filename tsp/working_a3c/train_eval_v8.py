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
        self.local_network = get_vqvae(num_cities+1, num_cities)  # Assuming correct model creation

    def select_action(self, state, epsilon=0.0, epsilon_decay=0.0):
        # Assuming state is shaped correctly as (1, num_cities + 1)
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        policy_logits, _ = self.local_network(state)

        if np.random.rand() < epsilon:
            # Random action (exploration)
            return np.random.choice(self.num_cities)
        else:
            # Greedy action (exploitation)
            return np.random.choice(self.num_cities, p=np.squeeze(policy_logits))

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

            # # Print intermediate values for debugging
            # print(f"Advantages: {advantages.numpy()}")
            # print(f"Returns: {returns.numpy()}")
            # print(f"Values: {values.numpy()}")

            # Compute policy loss
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=actions)
            policy_loss = tf.reduce_mean(advantages * neg_log_prob)

            # Compute value loss
            value_loss = tf.reduce_mean(tf.square(advantages))

            total_loss = policy_loss + 0.5 * value_loss
        
        grads = tape.gradient(total_loss, self.local_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.local_network.trainable_variables))

        # # Print loss components for debugging
        # print(f"Policy Loss: {policy_loss.numpy()}")
        # print(f"Value Loss: {value_loss.numpy()}")
        # print(f"Total Loss: {total_loss.numpy()}")

        return total_loss


# Training loop
def train_a3c(env, global_network, optimizer, max_episodes=1000):
    agent = A3CAgent(env.num_cities, global_network, optimizer)
    max_steps_per_episode = 100

    # Inside train_a3c function, adjust epsilon and epsilon_decay
    epsilon = 1.0  # Initial exploration rate
    epsilon_decay = 0.995  # Decay rate per episode

    for episode in range(max_episodes):
        state = env.reset()
        states, actions, rewards = [], [], []
        done = False

        while not done:
            action = agent.select_action(state, epsilon, epsilon_decay)  # Exclude the start city
            epsilon *= epsilon_decay
            next_state, reward, done, _ = env.step(action)

            states.append(state.copy())  # (current_city, visited_roads)
            actions.append(action)
            rewards.append(reward)

            # if done:
            #     break

            state = next_state

        loss = agent.compute_loss(states, actions, rewards)
        print(f"Episode {episode + 1}/{max_episodes}, Loss: {loss.numpy()}")

        # Update global network periodically (optional)
        global_network.set_weights(agent.local_network.get_weights())


# Example usage
# Define environment and global network
env = TSPEnv(cities)
global_network = get_vqvae(env.num_cities+1, env.num_cities)
optimizer = Adam(learning_rate=0.0001, clipvalue=1.0)

# Train A3C
train_a3c(env, global_network, optimizer, max_episodes=100)
