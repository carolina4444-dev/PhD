import tensorflow as tf
import numpy as np
import threading
#from model import ActorCriticModel
from model import get_vqvae


class A3C_Agent:
    def __init__(self, global_model, num_actions, gamma=0.99):
        self.global_model = global_model
        self.local_model = get_vqvae(num_actions+1, num_actions)
        self.local_model.build(input_shape=(None, num_actions + 1))  # Correct input shape
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.gamma = gamma
        self.num_actions = num_actions

    def train(self, env, max_steps_per_episode=1000, max_episodes=100):
        best_path = None
        best_reward = float('-inf')
        
        for episode in range(max_episodes):
            state = env.reset()
            current_path = []
            episode_reward = 0
            total_distance = 0
            current_city = state[0]
            current_path.append(current_city)

            with tf.GradientTape(persistent=True) as tape:
                for t in range(max_steps_per_episode):
                    # Combine current city and visited cities into a single array
                    current_city = state[0]
                    visited_cities = np.array(state[1], dtype=np.float32)
                    state_input = np.concatenate(([current_city], visited_cities), axis=None).astype(np.float32)
                    state_input = tf.convert_to_tensor(state_input)
                    state_input = tf.expand_dims(state_input, 0)

                    policy, value = self.local_model(state_input)
                    action = np.random.choice(self.num_actions, p=np.squeeze(policy))

                    next_state, reward, done, _ = env.step(action)
                    total_distance += reward  # Assuming reward is the negative distance
                    episode_reward += reward
                    current_path.append(next_state[0])

                    next_current_city = next_state[0]
                    next_visited_cities = np.array(next_state[1], dtype=np.float32)
                    next_state_input = np.concatenate(([next_current_city], next_visited_cities), axis=None).astype(np.float32)
                    next_state_input = tf.convert_to_tensor(next_state_input)
                    next_state_input = tf.expand_dims(next_state_input, 0)

                    _, next_value = self.local_model(next_state_input)

                    if done:
                        if all(next_state[1]) and next_state[0] == state[0]:  # All cities visited and returned to start
                            episode_reward += 150  # Add bonus for visiting all cities and returning to start
                        else:
                            episode_reward -= 50  # Penalty for not completing the tour

                        # Update best path and reward if this episode is better
                        if episode_reward > best_reward:
                            best_reward = episode_reward
                            best_path = current_path
                        
                        break

                    target = reward + self.gamma * next_value[0] if not done else reward
                    advantage = target - value[0]

                    actor_loss = -tf.math.log(policy[0, action]) * advantage
                    critic_loss = advantage**2
                    total_loss = actor_loss + critic_loss

                    grads = tape.gradient(total_loss, self.local_model.trainable_weights)
                    self.optimizer.apply_gradients(zip(grads, self.global_model.trainable_weights))

                    state = next_state

            print(f"Episode {episode + 1}/{max_episodes}, Episode reward: {episode_reward}")
            del tape  # Release resources

        return best_path, best_reward




# class A3C_Agent:
#     def __init__(self, global_model, num_actions, gamma=0.99):
#         self.global_model = global_model
#         #self.local_model = ActorCriticModel(num_actions)
#         self.local_model = get_vqvae(num_actions+1, num_actions)
#         self.local_model.build(input_shape=(None, num_actions + 1))  # Correct input shape
#         self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
#         self.gamma = gamma
#         self.num_actions = num_actions

#     def train(self, env, max_steps_per_episode=1000):
#         while True:
#             state = env.reset()
#             episode_reward = 0
#             with tf.GradientTape(persistent=True) as tape:
#                 for t in range(max_steps_per_episode):
#                     # Combine current city and visited cities into a single array
#                     state_input = np.concatenate(([state[0]], state[1]), axis=None).astype(np.float32)
#                     state_input = tf.convert_to_tensor(state_input)
#                     state_input = tf.expand_dims(state_input, 0)

#                     policy, value = self.local_model(state_input)
#                     action = np.random.choice(self.num_actions, p=np.squeeze(policy))

#                     next_state, reward, done, _ = env.step(action)
#                     episode_reward += reward

#                     # Combine next current city and visited cities into a single array
#                     next_state_input = np.concatenate(([next_state[0]], next_state[1]), axis=None).astype(np.float32)
#                     next_state_input = tf.convert_to_tensor(next_state_input)
#                     next_state_input = tf.expand_dims(next_state_input, 0)

#                     _, next_value = self.local_model(next_state_input)

#                     if done:
#                         target = reward
#                     else:
#                         target = reward + self.gamma * next_value[0]

#                     advantage = target - value[0]

#                     actor_loss = -tf.math.log(policy[0, action]) * advantage
#                     critic_loss = advantage**2
#                     total_loss = actor_loss + critic_loss

#                     grads = tape.gradient(total_loss, self.local_model.trainable_weights)
#                     self.optimizer.apply_gradients(zip(grads, self.global_model.trainable_weights))

#                     state = next_state

#                     if done:
#                         break
#             print("Episode reward: {}".format(episode_reward))

