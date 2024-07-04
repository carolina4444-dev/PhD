import numpy as np
from tsp_env import TSPEnv
import tensorflow as tf
from collections import deque
from dqnet_model import create_q_network

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

# Create the environment
env = TSPEnv(cities)

# Define constants
NUM_CITIES = len(cities)  # Number of cities (nodes)
NUM_EPISODES = 1000  # Number of episodes for training
MAX_STEPS = NUM_CITIES  # Maximum steps per episode (equal to number of cities)
BATCH_SIZE = 32  # Mini-batch size for training
GAMMA = 0.99  # Discount factor
EPSILON_START = 1.0  # Starting value of epsilon (for epsilon-greedy)
EPSILON_MIN = 0.01  # Minimum value of epsilon
EPSILON_DECAY = 0.995  # Decay rate of epsilon per episode
LEARNING_RATE = 0.001

# Define the replay buffer
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in idx]
    
    def size(self):
        return len(self.buffer)

# Initialize the environment and replay buffer
# Assume you have functions defined for environment initialization and step execution

# Initialize the DQN model
state_dim = NUM_CITIES+1  # State representation dimension (current city and visited cities)
action_dim = NUM_CITIES  # Number of possible actions (next city to visit)

model = create_q_network((state_dim), action_dim)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.Huber())

target_model = create_q_network((state_dim), action_dim)
target_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.Huber())
target_model.set_weights(model.get_weights())  # Initialize target model weights

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

# Initialize replay buffer
replay_buffer = ReplayBuffer(max_size=10000)

rewards_dqnet = []
episode_rewards = []

# Training loop
epsilon = EPSILON_START
for episode in range(1, NUM_EPISODES + 1):
    state = env.reset()  # Initialize the environment (reset to initial state)
    state = np.reshape(state, [1, state_dim])
    total_reward = 0
    episode_rewards = []
    
    for step in range(MAX_STEPS):
        # Epsilon-greedy policy
        if np.random.rand() <= epsilon:
            action = np.random.randint(action_dim)  # Explore: choose random action
        else:
            q_values = model.predict(state)
            action = np.argmax(q_values[0])  # Exploit: choose action with max Q-value
        
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_dim])
        total_reward += reward
        
        # Store experience in replay buffer
        replay_buffer.add((state, action, reward, next_state, done))
        
        state = next_state  # Move to next state
        
        # Perform mini-batch gradient descent
        if replay_buffer.size() >= BATCH_SIZE:
            minibatch = replay_buffer.sample(BATCH_SIZE)
            states, actions, rewards, next_states, dones = zip(*minibatch)
            states = np.squeeze(states)
            next_states = np.squeeze(next_states)
            actions = np.squeeze(actions)
            rewards = np.squeeze(rewards)
            dones = np.squeeze(dones)
            
            with tf.GradientTape() as tape:
                q_values = model(states)
                next_q_values = target_model(next_states)
                
                q_target = rewards + GAMMA * tf.reduce_max(next_q_values, axis=1) * (1 - dones)
                q_values_action = tf.reduce_sum(q_values * tf.one_hot(actions, action_dim), axis=1)
                
                loss = tf.reduce_mean(tf.square(q_target - q_values_action))
            
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        if done:
            break

        episode_rewards.append(reward)
    
    # Update target network periodically
    if episode % 10 == 0:
        target_model.set_weights(model.get_weights())
    
    # Decay epsilon
    if epsilon > EPSILON_MIN:
        epsilon *= EPSILON_DECAY

    rewards_dqnet.extend(episode_rewards)
    
    # Print episode information
    print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {epsilon}")

# After training, you can evaluate the trained model by running episodes with a deterministic policy
# and measuring performance metrics such as total distance traveled.
import pickle
# Step 2: Pickle the list of arrays to a file
with open('rewards_dqnet.pkl', 'wb') as f:
    pickle.dump(rewards_dqnet, f)

print("List of arrays has been pickled to 'rewards_dqnet.pkl'")