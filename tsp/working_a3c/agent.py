import tensorflow as tf
import numpy as np
from model import get_vqvae
import numpy as np
from tsp_env import TSPEnv  # Assuming this is your custom TSP environment class
from model import get_vqvae  # Assuming this function creates your model correctly

class A3C_Agent:
    def __init__(self, num_actions, env, gamma=0.99):
        self.local_model = get_vqvae(num_actions + 1, num_actions)  # Correct input shape
        self.local_model.build(input_shape=(None, num_actions))

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.gamma = gamma
        self.num_actions = num_actions
        self.env = env
        
        self.episode_reward = 0
        self.current_path = [env.start_city]
        self.state = env.reset()

    def train_step(self, state):
        with tf.GradientTape() as tape:
            current_city = state[0]
            visited_cities = np.array(state[1], dtype=np.float32)
            state_input = np.concatenate(([current_city], visited_cities)).astype(np.float32)
            state_input = tf.expand_dims(state_input, 0)

            # Get policy and value from the model
            policy, value = self.local_model(state_input, training=True)
            action = np.random.choice(self.num_actions, p=np.squeeze(policy))

            # Perform the action in the environment
            next_state, reward, done, _ = self.env.step(action)
            self.episode_reward += reward
            self.current_path.append(next_state[0])

            # Prepare the next state input for the model
            next_current_city = next_state[0]
            next_visited_cities = np.array(next_state[1], dtype=np.float32)
            next_state_input = np.concatenate(([next_current_city], next_visited_cities)).astype(np.float32)
            next_state_input = tf.expand_dims(next_state_input, 0)

            _, next_value = self.local_model(next_state_input)
            # Compute target and advantage
            target = reward + self.gamma * next_value[0] if not done else reward
            advantage = target - value[0]

            # Compute losses
            actor_loss = -tf.math.log(policy[0, action]) * advantage
            critic_loss = tf.square(advantage)
            total_loss = actor_loss + critic_loss

        # Compute and apply gradients
        grads = tape.gradient(total_loss, self.local_model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.local_model.trainable_weights))

        if done:
            if all(next_state[1]) and next_state[0] == self.env.start_city:  # All cities visited and returned to start
                self.episode_reward += 50  # Bonus for completing the tour
            else:
                self.episode_reward -= 50  # Penalty for not completing the tour

        # Update the current state
        self.state = next_state

        # Optionally render the environment
        self.env.render()

        return self.current_path, self.episode_reward

    def train(self, num_episodes=1000):


        best_reward = -float('inf')
        best_path = None

        for episode in range(num_episodes):
            episode_path, episode_reward = self.train_step(self.state)
            
            # Check if this episode has a better reward than the current best
            if episode_reward > best_reward:
                best_reward = episode_reward
                best_path = episode_path.copy()

            if episode % 100 == 0:
                print(f"Episode {episode}: Reward = {episode_reward}")

        return best_path, best_reward

