import numpy as np
from tsp_env import TSPEnv
from model import get_vqvae  # Assuming this function creates your model correctly
import tensorflow as tf
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

# Initialize agent parameters
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
gamma = 0.99

# Epsilon-greedy parameters
epsilon = 1.0  # Initial epsilon value
epsilon_min = 0.01  # Minimum epsilon value
epsilon_decay = 0.995  # Epsilon decay rate per episode

# Global variables for shared model and optimizer
global_model = get_vqvae(num_actions + 1, num_actions)  # Correct input shape
global_model.build(input_shape=(None, num_actions))
global_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Function to copy weights from global model to local model
def update_local_model(local_model, global_model):
    local_model.set_weights(global_model.get_weights())

# A3C agent class
class A3C_Agent:
    def __init__(self, model, optimizer, state_size, num_actions):
        self.local_model = model
        self.optimizer = optimizer
        self.state_size = state_size
        self.num_actions = num_actions
        self.env = TSPEnv(cities)

    def train(self, max_episodes=1000):
        global epsilon
        
        for episode in range(max_episodes):
            state = self.env.reset()
            episode_reward = 0
            memory = []
            done = False

            while not done:
                # Epsilon-greedy action selection
                if np.random.rand() < epsilon:
                    action = np.random.choice(self.num_actions)
                    was_random = True
                else:
                    current_city = np.array([state[0]], dtype=np.float32)
                    visited_cities = np.array(state[1:], dtype=np.float32)
                    state_input = np.concatenate((current_city, visited_cities)).astype(np.float32)
                    state_input = tf.expand_dims(state_input, 0)

                    # Get policy from the model
                    policy, _ = self.local_model(state_input)
                    action = np.random.choice(self.num_actions, p=np.squeeze(policy))

                    was_random = False

                # Perform the action in the environment
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward

                # Store the experience in memory
                memory.append((state, action, reward, next_state, done))
                state = next_state

                if done:
                    if all(next_state[1:]) and next_state[0] == self.env.start_city:
                        episode_reward += 50  # Bonus for completing the tour
                    else:
                        episode_reward -= 50  # Penalty for not completing the tour

                    env.render()
                    break

                # Update epsilon
                if epsilon > epsilon_min:
                    epsilon *= epsilon_decay

                env.render()

                if not was_random:
                    # Update local model and train
                    with tf.GradientTape() as tape:
                        current_city = np.array([state[0]], dtype=np.float32)
                        visited_cities = np.array(state[1:], dtype=np.float32)
                        state_input = np.concatenate((current_city, visited_cities)).astype(np.float32)
                        state_input = tf.expand_dims(state_input, 0)

                        # Get policy and value from the model
                        policy, value = self.local_model(state_input)

                        # Compute target and advantage
                        next_state_tensor = tf.convert_to_tensor([next_state], dtype=tf.float32)
                        _, next_value = self.local_model(next_state_tensor)
                        target = reward + gamma * next_value[0][0] if not done else reward
                        advantage = target - value[0][0]

                        # Compute losses
                        actor_loss = -tf.math.log(policy[0, action]) * advantage
                        critic_loss = tf.square(advantage)
                        total_loss = actor_loss + critic_loss

                    # Compute gradients
                    grads = tape.gradient(total_loss, self.local_model.trainable_weights)

                    # Apply gradients to global model
                    self.optimizer.apply_gradients(zip(grads, self.local_model.trainable_weights))

                # Sync local model with global model periodically
                if episode % 5 == 0:
                    update_local_model(self.local_model, global_model)

            print(f"Episode {episode}: Reward = {episode_reward}, Epsilon = {epsilon}")

# Initialize and run A3C agent
agent = A3C_Agent(global_model, global_optimizer, state_size, num_actions)
agent.train(max_episodes=10) #1000
