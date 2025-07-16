import matplotlib.pyplot as plt
import numpy as np
import math

# Set up the plot
plt.figure(figsize=(20, 16))
plt.xlabel('Steps')
plt.ylabel('Average Reward')

# Get user inputs
# Input standard deviation (if you have a variance instead, you can compute std = math.sqrt(variance_input))
std = float(input('Enter standard deviation (for reward noise): '))
machines = int(input('Number of machines: '))
practice = int(input('Number of practice spins: '))
spins = int(input('Total spins: '))

# Initialize the true mean for each machine (sampled from a standard normal)
BanditMeans = [np.random.normal(0, 1) for _ in range(machines)]

all_averages = []
best_machine_avg = -float('inf')  # Start with -infinity so that even negative averages count
best_machine = 0
best_avg = []  # This will hold the cumulative averages for the best machine

# Evaluate each machine during practice spins
for j in range(machines):
    rewards = []
    averages = []
    for i in range(practice):
        reward = np.random.normal(BanditMeans[j], std)
        rewards.append(reward)
        avg = np.mean(rewards)
        averages.append(avg)
    
    # Choose the machine with the highest last average from practice spins
    if averages[-1] > best_machine_avg:
        best_machine_avg = averages[-1]
        best_machine = j
        best_avg = averages.copy()
    
    all_averages.append(averages)

# Continue spinning for the best machine (stationary environment)
for i in range(spins - practice):
    reward = np.random.normal(BanditMeans[best_machine], std)
    # Update the cumulative average for the best machine using the previous average and the new reward
    new_count = len(best_avg) + 1
    new_avg = (best_avg[-1] * len(best_avg) + reward) / new_count
    best_avg.append(new_avg)

# Replace the best machine's record in all_averages with the updated values
all_averages[best_machine] = best_avg

# Pad each machine's list with NaN so that all lists have equal length for plotting
max_length = max(len(avg) for avg in all_averages)
all_averages_padded = [avg + [np.nan] * (max_length - len(avg)) for avg in all_averages]
all_averages_padded = np.array(all_averages_padded, dtype=float)

# Compute the overall cumulative average across machines
overall_cumulative_avg = []
cumulative_total = 0
cumulative_count = 0

for i in range(max_length):
    if i < practice:
        valid_values = [all_averages_padded[j][i] for j in range(machines) if not np.isnan(all_averages_padded[j][i])]
        cumulative_total += sum(valid_values)
        cumulative_count += len(valid_values)
    else:
        valid_value = all_averages_padded[best_machine][i]
        if not np.isnan(valid_value):
            cumulative_total += valid_value
            cumulative_count += 1
    overall_cumulative_avg.append(cumulative_total / cumulative_count)

# Plot the results
for j in range(machines):
    if j != best_machine:
        # For non-best machines, plot only the practice spins
        plt.plot(all_averages_padded[j][:practice], label=f'Machine {j + 1}', linewidth=1)
    else:
        # Plot the full series for the best machine
        plt.plot(all_averages_padded[j], label=f'Best Machine {j + 1}', linestyle='--', color='blue')

plt.plot(overall_cumulative_avg, color='black', label='Overall Cumulative Average', linewidth=2)
plt.legend()
plt.show()

