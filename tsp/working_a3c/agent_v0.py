import tensorflow as tf
import numpy as np
from model import get_vqvae

class A3C_Agent:
    def __init__(self, global_model, num_actions, gamma=0.99):
        self.global_model = global_model
        self.local_model = get_vqvae(num_actions + 1, num_actions)  # Correct input shape
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.gamma = gamma
        self.num_actions = num_actions

    def train(self, env, max_steps_per_episode=100, max_episodes=100):
        best_path = None
        best_reward = float('-inf')

        for episode in range(max_episodes):
            state = env.reset()
            current_path = [env.start_city]
            episode_reward = 0

            with tf.GradientTape() as tape:
                for t in range(max_steps_per_episode):
                    # Prepare the state input for the model
                    current_city = state[0]
                    visited_cities = np.array(state[1], dtype=np.float32)
                    state_input = np.concatenate(([current_city], visited_cities)).astype(np.float32)
                    state_input = tf.expand_dims(state_input, 0)

                    # Get policy and value from the model
                    policy, value = self.local_model(state_input)
                    action = np.random.choice(self.num_actions, p=np.squeeze(policy))

                    # Perform the action in the environment
                    next_state, reward, done, _ = env.step(action)
                    episode_reward += reward
                    current_path.append(next_state[0])

                    # Prepare the next state input for the model
                    next_current_city = next_state[0]
                    next_visited_cities = np.array(next_state[1], dtype=np.float32)
                    next_state_input = np.concatenate(([next_current_city], next_visited_cities)).astype(np.float32)
                    next_state_input = tf.expand_dims(next_state_input, 0)

                    _, next_value = self.local_model(next_state_input)

                    if done:
                        if all(next_state[1]) and next_state[0] == env.start_city:  # All cities visited and returned to start
                            episode_reward += 50  # Bonus for completing the tour
                        else:
                            episode_reward -= 50  # Penalty for not completing the tour

                        # Update the best path and reward if this episode is better
                        if episode_reward > best_reward:
                            best_reward = episode_reward
                            best_path = current_path
                        
                        break

                    # Compute target and advantage
                    target = reward + self.gamma * next_value[0] if not done else reward
                    advantage = target - value[0]

                    # Compute losses
                    actor_loss = -tf.math.log(policy[0, action]) * advantage
                    critic_loss = tf.square(advantage)
                    total_loss = actor_loss + critic_loss

                    # Compute and apply gradients
                    grads = tape.gradient(total_loss, self.local_model.trainable_weights)
                    self.optimizer.apply_gradients(zip(grads, self.global_model.trainable_weights))

                    # Update the current state
                    state = next_state

                    # Optionally render the environment
                    env.render()

            # Print episode summary
            print(f"Episode {episode + 1}/{max_episodes}, Episode reward: {episode_reward}")
            env.render()

        return best_path, best_reward
