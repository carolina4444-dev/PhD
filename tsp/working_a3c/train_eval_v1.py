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
    [1, 2]
]

# Create the environment
env = TSPEnv(cities)

# Calculate the number of actions as the number of cities
num_actions = len(cities)

# State size includes the current city and the visited cities
state_size = num_actions + 1

# Initialize agent
#agent = A3C_Agent(num_actions, env)

local_model = get_vqvae(num_actions + 1, num_actions)  # Correct input shape
local_model.build(input_shape=(None, num_actions))
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
gamma = 0.99






num_episodes = 1000

best_reward = -float('inf')
best_path = None


for episode in range(num_episodes):

    done = False
    state = env.reset()
    current_path = [env.start_city]
    episode_reward = 0

    while not done:
        # Train the agent
        
        with tf.GradientTape() as tape:
            current_city = np.array([state[0]], dtype=np.float32)
            visited_cities = np.array(state[1:], dtype=np.float32)
            state_input = np.concatenate((current_city, visited_cities)).astype(np.float32)
            state_input = tf.expand_dims(state_input, 0)

            # Get policy and value from the model
            policy, value = local_model(state_input)

            action = np.random.choice(num_actions, p=np.squeeze(policy))

            # Perform the action in the environment
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
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

            _, next_value = local_model(next_state_input)
            # Compute target and advantage
            target = reward + gamma * next_value[0] if not done else reward
            advantage = target - value[0]

            # Compute losses
            actor_loss = -tf.math.log(policy[0, action]) * advantage
            critic_loss = tf.square(advantage)
            total_loss = actor_loss + critic_loss

        # Compute and apply gradients
        grads = tape.gradient(total_loss, local_model.trainable_weights)
        optimizer.apply_gradients(zip(grads, local_model.trainable_weights))

        # Update the current state
        state = next_state

        # Optionally render the environment
        env.render()

    # Check if this episode has a better reward than the current best
    if episode_reward > best_reward:
        best_reward = episode_reward
        best_path = current_path.copy()

    if episode % 100 == 0:
        print(f"Episode {episode}: Reward = {episode_reward}")
    
    

env.render()


# # Example usage
# distance_matrix = np.array([
#     [0, 2, 9, 10],
#     [1, 0, 6, 4],
#     [15, 7, 0, 8],
#     [6, 3, 12, 0]
# ])
# env = TSPEnv(distance_matrix, start_city=0, end_city=0)
# obs = env.reset()
# done = False
# while not done:
#     action = env.action_space.sample()
#     obs, reward, done, info = env.step(action)
#     env.render()

