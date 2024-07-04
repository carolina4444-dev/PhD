import matplotlib.pyplot as plt
import pickle
import numpy as np

with open('rewards_dqnet.pkl', 'rb') as f:
    data = pickle.load(f)


def cumulative_sum(reward_array):
    total_reward_array = []
    cumulative_sum = 0
    for reward in reward_array:
        cumulative_sum += reward
        total_reward_array.append(cumulative_sum)
    return total_reward_array

# Calculate cumulative sums and cumulative counts
cumulative_sums = np.cumsum(data)  # cumulative sum up to each index
cumulative_counts = np.arange(1, len(data) + 1)  # cumulative count up to each index

# Calculate cumulative averages
cumulative_averages = cumulative_sums / cumulative_counts


# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(data, label='Episode Rewards', color='blue')
plt.plot(cumulative_averages, label='Average Rewards', color='red')

# Add titles and labels
plt.title('Rewards Chart for DQN Reinforcement Learning')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
