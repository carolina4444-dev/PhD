import matplotlib.pyplot as plt
import pickle
import numpy as np

with open('mean_reward_a3c_tsp.pkl', 'rb') as f:
    data = pickle.load(f)


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
plt.title('Rewards Chart for A3C Reinforcement Learning')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
