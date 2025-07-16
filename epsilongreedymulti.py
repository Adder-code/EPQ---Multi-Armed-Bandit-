import matplotlib.pyplot as plt
import numpy as np
import random

plt.figure(figsize=(20, 16))
plt.xlabel('Steps')
plt.ylabel('Cumulative Average Reward')

variance = 1 
machines = int(input('Machines - '))
spins = int(input('Spins - '))
epsilon = 10
BanditMeans = [np.random.normal(0, 1) for _ in range(machines)]
rewards = [0.0] * machines
pulls = [0] * machines
acr = []

cumulative_reward = 0.0  

for i in range(spins):
    if i % epsilon != 0:

        chosen_machine = np.argmax(rewards)
    else:

        chosen_machine = random.randint(0, len(rewards) - 1)


    reward = np.random.normal(BanditMeans[chosen_machine], variance)
    rewards[chosen_machine] += reward
    pulls[chosen_machine] += 1


    cumulative_reward += reward
    cumulative_avg_reward = cumulative_reward / (i + 1)


    acr.append(cumulative_avg_reward)


plt.plot(acr, label='Cumulative Average Reward')
plt.title('Cumulative Average Reward Over Spins')
plt.legend()
plt.show()
