import numpy as np
from tsp_env import TSPEnv
from model import get_vqvae  # Assuming this function creates your model correctly
import tensorflow as tf

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

# Initialize agent
local_model = get_vqvae(num_actions + 1, num_actions)  # Correct input shape
local_model.build(input_shape=(None, num_actions))
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
gamma = 0.99

# Epsilon-greedy parameters
epsilon = 1.0  # Initial epsilon value
epsilon_min = 0.01  # Minimum epsilon value
epsilon_decay = 0.995  # Epsilon decay rate per episode

num_episodes = 1000

best_reward = -float('inf')
best_path = None

for episode in range(num_episodes):
    done = False
    state = env.reset()
    current_path = [env.start_city]
    episode_reward = 0

    while not done:
        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action = np.random.choice(num_actions)
        else:
            current_city = np.array([state[0]], dtype=np.float32)
            visited_cities = np.array(state[1:], dtype=np.float32)
            state_input = np.concatenate((current_city, visited_cities)).astype(np.float32)
            state_input = tf.expand_dims(state_input, 0)

            # Get policy from the model
            policy, _ = local_model(state_input)
            action = np.argmax(policy[0])

        # Perform the action in the environment
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        current_path.append(next_state[0])

        # Prepare the next state input for the model
        next_current_city = np.array([next_state[0]])
        next_visited_cities = np.array(next_state[1:], dtype=np.float32)
        next_state_input = np.concatenate((next_current_city, next_visited_cities)).astype(np.float32)
        next_state_input = tf.expand_dims(next_state_input, 0)

        # Train the agent
        with tf.GradientTape() as tape:
            policy, value = local_model(next_state_input)
            target = reward + gamma * value[0] if not done else reward
            advantage = target  # As critic_loss is unused, advantage is equivalent to target

            # Compute actor loss
            actor_loss = -tf.math.log(policy[0, action]) * advantage
            total_loss = actor_loss

        # Compute and apply gradients
        grads = tape.gradient(total_loss, local_model.trainable_weights)
        optimizer.apply_gradients(zip(grads, local_model.trainable_weights))

        # Update epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        # Update the current state
        state = next_state

        env.render()

    # Check if this episode has a better reward than the current best
    if episode_reward > best_reward:
        best_reward = episode_reward
        best_path = current_path.copy()

    if episode % 100 == 0:
        print(f"Episode {episode}: Reward = {episode_reward}, Epsilon = {epsilon}")

print(f"Best Path: {best_path}")
print(f"Best Reward: {best_reward}")
