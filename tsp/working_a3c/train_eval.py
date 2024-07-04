import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from model import get_vqvae  # Assuming this function creates your model correctly

# Define cities as coordinates (example)
cities = [
    [23, 45],
    [57, 12],
    [38, 78],
    [92, 34],
    [45, 67],
    [18, 90],
    [72, 55],
    [66, 24],
    [83, 62],
    [49, 40]
]

# A3C Agent
class A3CAgent:
    def __init__(self, num_cities, optimizer, gamma=0.99):
        self.num_cities = num_cities
        self.optimizer = optimizer
        self.gamma = gamma
        self.local_network = get_vqvae(num_cities + 1, num_cities)  # Including current city in input size

    def select_action(self, state, visited, epsilon=0.0):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        policy, critic_value = self.local_network(state)

        policy = policy.numpy()[0]
        #policy = tf.nn.softmax(policy_logits).numpy()[0]

        # Apply mask to prevent revisiting cities
        mask = np.array(visited, dtype=np.float32)
        masked_policy = policy * (1 - mask)
        masked_policy /= masked_policy.sum()

        if np.random.rand() < epsilon:
            # Random action (exploration)
            action = np.random.choice(self.num_cities, p=masked_policy)
        else:
            # Greedy action (exploitation)
            action = np.argmax(masked_policy)
        
        return action, masked_policy, critic_value

    def compute_loss(self, action_probs_history, critic_value_history, rewards, next_critic_value):
        # Convert history lists to TensorFlow tensors
        action_probs_history = tf.convert_to_tensor(action_probs_history, dtype=tf.float32)
        critic_value_history = tf.convert_to_tensor(critic_value_history, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)

        # Calculate advantages (A(t) in A3C)
        deltas = rewards + self.gamma * next_critic_value - critic_value_history

        # Compute actor (policy) loss
        actor_loss = -tf.math.log(action_probs_history) * tf.stop_gradient(deltas)
        actor_loss = tf.reduce_mean(actor_loss)

        # Compute critic loss (mean squared error)
        critic_loss = tf.reduce_mean(tf.square(deltas))

        # Entropy regularization to encourage exploration
        entropy = -tf.reduce_sum(action_probs_history * tf.math.log(action_probs_history + 1e-20), axis=-1)
        entropy_loss = -0.01 * tf.reduce_mean(entropy)  # 0.01 is a hyperparameter, can be adjusted

        # Total loss is a combination of actor and critic losses
        total_loss = actor_loss + critic_loss + entropy_loss

        return total_loss

# Environment class with reward calculation fixed
class TSPEnv:
    def __init__(self, cities):
        self.cities = cities
        self.num_cities = len(cities)
        self.distance_matrix = self.compute_distance_matrix(cities)
        self.max_distance = np.max(self.distance_matrix)
        self.reset()

    def compute_distance_matrix(self, cities):
        num_cities = len(cities)
        distance_matrix = np.zeros((num_cities, num_cities))
        for i in range(num_cities):
            for j in range(num_cities):
                distance_matrix[i][j] = np.linalg.norm(np.array(cities[i]) - np.array(cities[j]))
        return distance_matrix

    def reset(self):
        self.current_city = 0
        self.visited = [False] * self.num_cities
        self.visited[self.current_city] = True
        self.total_distance = 0
        self.num_visited = 1
        return self.get_state()

    def get_state(self):
        return [self.current_city] + self.visited

    def step(self, action):
        if self.visited[action]:
            # Penalty for revisiting a city
            reward = -100
            done = True
        else:
            # Calculate normalized reward
            distance = self.distance_matrix[self.current_city][action]
            normalized_reward = 100 * (1 - (distance / self.max_distance))
            reward = normalized_reward
            self.total_distance += distance
            self.current_city = action
            self.visited[action] = True
            self.num_visited += 1
            done = self.num_visited == self.num_cities

        return self.get_state(), reward, done, {}

    def render(self, mode='human'):
        print("Current city:", self.current_city)
        print("Visited cities:", self.visited)
        print("Total distance:", self.total_distance)

# Training loop
def train_a3c(env, optimizer, max_episodes):
    agent = A3CAgent(env.num_cities, optimizer)
    epsilon = 1.0  # Initial exploration parameter
    epsilon_min = 0.01  # Minimum exploration parameter
    epsilon_decay = 0.995  # Decay rate for exploration parameter
    max_steps_per_episode = env.num_cities

    best_road = None
    best_total_distance = float('inf')

    total_reward = []

    for episode in range(max_episodes):
        state = env.reset()
        visited_cities = [False] * env.num_cities
        visited_cities[state[0]] = True  # Mark the starting city as visited
        road_tracking = [state[0]]

        states, actions, rewards = [], [], []

        action_probs_history = []
        critic_history = []

        done = False
        with tf.GradientTape() as tape:
            for step in range(max_steps_per_episode):
                action, action_probs, critic_value = agent.select_action(state, visited_cities, epsilon)

                next_state, reward, done, _ = env.step(action)
                
                # Track road
                road_tracking.append(next_state[0])
                visited_cities[next_state[0]] = True

                states.append(state.copy())  # Include current city and visited cities
                actions.append(action)
                rewards.append(reward)

                action_probs_history.append(action_probs[action])
                critic_history.append(critic_value[0, 0])

                # Update state
                state = next_state

                if done:
                    break

            # Calculate the value of the next state
            if not done:
                next_state_tensor = tf.convert_to_tensor([next_state], dtype=tf.float32)
                next_action_prob, next_critic_value = agent.local_network.predict(next_state_tensor)
                next_critic_value = next_critic_value[0, 0]
            else:
                next_critic_value = 0

            loss = agent.compute_loss(action_probs_history, critic_history, rewards, next_critic_value)

            total_reward.append(reward)
        
        # Backpropagation
        grads = tape.gradient(loss, agent.local_network.trainable_variables)
        agent.optimizer.apply_gradients(zip(grads, agent.local_network.trainable_variables))

        # Update the best road found so far
        if env.total_distance < best_total_distance and all(visited_cities):
            best_total_distance = env.total_distance
            best_road = road_tracking.copy()

        print(f"Episode {episode + 1}/{max_episodes}, Loss: {loss.numpy()}, Total Distance: {env.total_distance}")
        print(f"Visited cities: {visited_cities}")
        print(f"Road tracking: {road_tracking}")

        # Decay epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

    print(f"Best Road: {best_road}, Best Total Distance: {best_total_distance}")

    return total_reward

# Example usage
env = TSPEnv(cities)
optimizer = Adam(learning_rate=0.0001)

# Train A3C
total_reward = train_a3c(env, optimizer, max_episodes=10000)
env.render()

import pickle
# Step 2: Pickle the list of arrays to a file
with open('mean_reward_a3c_tsp.pkl', 'wb') as f:
    pickle.dump(total_reward, f)

print("List of arrays has been pickled to 'mean_reward_a3c_tsp.pkl'")
