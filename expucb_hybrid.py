import numpy as np
import math
import random

def hybrid_exponential_ucb(machines, spins, BanditMeans):
    epsilon_0 = 1.0  # Initial exploration rate
    decay_rate = 0.01  # Exponential decay rate
    rewards = [0.0] * machines
    counts = [0] * machines
    acr = []
    cumulative_reward = 0.0

    # Initializing each machine once (ensuring UCB starts with valid values)
    for i in range(machines):
        reward = np.random.normal(BanditMeans[i], 1)
        rewards[i] += reward
        counts[i] += 1
        cumulative_reward += reward

    # Main loop
    for i in range(machines, spins):
        epsilon = epsilon_0 * math.exp(-decay_rate * i)  # Decaying exploration rate

        if random.random() < epsilon:
            # Exploration: Choose a random machine
            selected_machine = random.randint(0, machines - 1)
        else:
            # Exploitation: Use UCB formula
            upper_bound_all = []
            for j in range(machines):
                avg_reward = rewards[j] / counts[j]
                exploration_term = math.sqrt((2 * math.log(i + 1)) / counts[j])
                upper_bound = avg_reward + exploration_term
                upper_bound_all.append(upper_bound)

            selected_machine = np.argmax(upper_bound_all)

        # Simulate reward
        reward = np.random.normal(BanditMeans[selected_machine], 1)
        rewards[selected_machine] += reward
        counts[selected_machine] += 1
        cumulative_reward += reward

        # Update cumulative average reward
        cumulative_avg_reward = cumulative_reward / (i + 1)
        acr.append(cumulative_avg_reward)

    return acr

machines = 10
spins = 1000
BanditMeans = np.random.rand(machines) * 10  

acr = hybrid_exponential_ucb(machines, spins, BanditMeans)

import matplotlib.pyplot as plt
plt.plot(acr)
plt.xlabel("Spins")
plt.ylabel("Average Cumulative Reward")
plt.title("Hybrid Exponential Decay + UCB Performance")
plt.show()
