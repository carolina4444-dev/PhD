import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from model import get_vqvae  # Assuming this function creates your model correctly

# Define cities as coordinates (example)
cities = [
    [0, 0],
    [1, 1],
    [2, 0],
    [1, 2],
    [0, 1],
    [2, 1],
    [1, 0],
    [0, 2]
]

# A3C Agent
class A3CAgent:
    def __init__(self, num_cities, optimizer, gamma=0.99):
        self.num_cities = num_cities
        self.optimizer = optimizer
        self.gamma = gamma
        self.local_network = get_vqvae(num_cities + 1, num_cities)  # Including current city in input size

    def select_action(self, state, epsilon=0.0):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        policy_logits, critic_value = self.local_network(state)
        action_probs = tf.nn.softmax(policy_logits)

        if np.random.rand() < epsilon:
            # Random action (exploration)
            action = np.random.choice(self.num_cities)
        else:
            # Greedy action (exploitation)
            action = np.random.choice(self.num_cities, p=np.squeeze(action_probs))
        
        return action, action_probs, critic_value

    def compute_loss(self, action_probs_history, critic_value_history, rewards, next_critic_value, action_probs_next, tape):
        gamma = 0.99
        eps = np.finfo(np.float32).eps.item()

        # Calculate expected value from rewards
        returns = []
        discounted_sum = next_critic_value
        for r in rewards[::-1]:
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)

        # Normalize
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
        returns = returns.tolist()

        # Calculating loss values to update our network
        actor_losses = []
        critic_losses = []
        for log_prob, value, ret in zip(action_probs_history, critic_value_history, returns):
            advantage = ret - value
            actor_losses.append(-tf.math.log(log_prob) * advantage)
            critic_losses.append(advantage**2)

        # Backpropagation
        loss_value = sum(actor_losses) + sum(critic_losses)
        grads = tape.gradient(loss_value, self.local_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.local_network.trainable_variables))
        return loss_value

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
            reward = 0
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

    mean_reward = []

    for episode in range(max_episodes):
        state = env.reset()
        visited_cities = [False] * env.num_cities
        visited_cities[state[0]] = True  # Mark the starting city as visited
        road_tracking = [state[0]]

        states, actions, rewards = [], [], []

        action_probs_history = []
        critic_history = []

        action_probs_next = []
        critic_next = []
        
        done = False
        with tf.GradientTape() as tape:
            for step in range(max_steps_per_episode):
                action, action_probs, critic_value = agent.select_action(state, epsilon)

                # Debugging statements
                print(f"Episode {episode + 1}, Step {step + 1}")
                print(f"State: {state}")
                print(f"Action: {action}")
                print(f"Action probabilities: {action_probs.numpy()}")
                print(f"Critic value: {critic_value.numpy()}")

                if action >= len(action_probs) or action < 0:
                    print(f"Invalid action: {action} for action_probs length: {len(action_probs)}")
                    break

                next_state, reward, done, _ = env.step(action)
                
                # Track road
                road_tracking.append(next_state[0])
                visited_cities[next_state[0]] = True

                states.append(state.copy())  # Include current city and visited cities
                actions.append(action)
                rewards.append(reward)

                action_probs_history.append(action_probs[0, action])
                critic_history.append(critic_value[0, 0])

                # Get the value of the next state
                next_state_tensor = tf.convert_to_tensor([next_state], dtype=tf.float32)
                next_action_prob, next_critic_value = agent.local_network(next_state_tensor)
                next_critic_value = next_critic_value[0, 0]
                critic_next.append(next_critic_value)
                action_probs_next.append(next_action_prob)

                state = next_state

                if done:
                    break

            loss = agent.compute_loss(action_probs_history, critic_history, rewards, critic_next, action_probs_next, tape)

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

        mean_reward.append(np.mean(rewards))
        print('Total reward:', mean_reward[-1])

    print(f"Best Road: {best_road}, Best Total Distance: {best_total_distance}")

    return mean_reward

# Example usage
env = TSPEnv(cities)
optimizer = Adam(learning_rate=0.0001)

# Train A3C
mean_reward = train_a3c(env, optimizer, max_episodes=10000)
env.render()

import pickle
# Step 2: Pickle the list of arrays to a file
with open('mean_reward_a3c_tsp.pkl', 'wb') as f:
    pickle.dump(mean_reward, f)

print("List of arrays has been pickled to 'mean_reward_a3c_tsp.pkl'")
