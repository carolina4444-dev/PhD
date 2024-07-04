import numpy as np
from tsp_env import TSPEnv
from model import get_vqvae  # Assuming this function creates your model correctly
import tensorflow as tf

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
local_model = get_vqvae(num_actions + 1, num_actions)  # Correct input shape
local_model.build(input_shape=(None, num_actions))

# Training parameters
num_episodes = 1000
best_reward = -float('inf')
best_path = None

# Training loop
for episode in range(num_episodes):
    state = env.reset()  # Reset environment for each episode
    current_path = [env.start_city]
    episode_reward = 0
    done = False

    while not done:
        with tf.GradientTape() as tape:
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = np.random.choice(num_actions)
                use_value = False  # Don't use value during exploration
            else:
                current_city = np.array([state[0]], dtype=np.float32)
                visited_cities = np.array(state[1:], dtype=np.float32)
                state_input = np.concatenate((current_city, visited_cities)).astype(np.float32)
                state_input = tf.expand_dims(state_input, 0)

                # Get policy and value from the model
                policy, value = local_model(state_input)
                action = np.random.choice(num_actions, p=np.squeeze(policy))
                use_value = True  # Use value during exploitation

            # Perform the action in the environment
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            # Update state
            state[0] = next_state[0]  # Update current city
            state[action + 1] = 1  # Mark the chosen city as visited

            # Update current path
            current_path.append(next_state[0])

            if done:
                if all(next_state[1:]) and next_state[0] == env.start_city:  # All cities visited and returned to start
                    episode_reward += 50  # Bonus for completing the tour
                else:
                    episode_reward -= 50  # Penalty for not completing the tour
                
                env.render()
                break

            # Prepare the next state input for the model
            next_current_city = np.array([next_state[0]])
            next_visited_cities = np.array(next_state[1:], dtype=np.float32)
            next_state_input = np.concatenate((next_current_city, next_visited_cities)).astype(np.float32)
            next_state_input = tf.expand_dims(next_state_input, 0)

            if use_value:
                _, next_value = local_model(next_state_input)
                # Compute target and advantage
                target = reward + gamma * next_value[0][0] if not done else reward
                advantage = target - value[0][0]

                # Compute losses
                actor_loss = -tf.math.log(policy[0, action]) * advantage
                critic_loss = tf.square(advantage)
                total_loss = actor_loss + critic_loss

                # Compute gradients and apply to the model
                grads = tape.gradient(total_loss, local_model.trainable_weights)
                optimizer.apply_gradients(zip(grads, local_model.trainable_weights))
        
        # # Update the current state for the next iteration
        # state = next_state

        # Decay epsilon after each episode
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        # Optionally render the environment
        env.render()

    # Check if this episode has a better reward than the current best
    if episode_reward > best_reward:
        best_reward = episode_reward
        best_path = current_path.copy()

    if episode % 100 == 0:
        print(f"Episode {episode}: Reward = {episode_reward}, Epsilon = {epsilon}")

print(f"Best Path: {best_path}")
print(f"Best Reward: {best_reward}")
