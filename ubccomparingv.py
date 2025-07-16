import matplotlib.pyplot as plt
import numpy as np
import math

plt.figure(figsize=(20, 16))
plt.xlabel('Steps')
plt.ylabel('Average Reward')

variance = int(input('variance - '))
machines = int(input('machines - '))
spins = int(input('spins - '))
constant = 0

while constant!=10:
    counts = [0] * machines  
    rewards = [0.0] * machines  
    lower_bound_all = []
    acr = []

    BanditMeans = [np.random.normal(0, 1) for _ in range(machines)]

    for j in range(spins):

        upper_bound_all = []
        lower_bound_all = []

        for i in range(machines):
            if counts[i] == 0:
                reward = np.random.normal(BanditMeans[i], variance)
                counts[i] += 1
            else:
                reward = np.random.normal(BanditMeans[i], variance)

            rewards[i] += reward  
            upper_bound = rewards[i] / counts[i] + constant * math.sqrt((2 * math.log(j + 1)) / counts[i])
            lower_bound = rewards[i] / counts[i] - constant * math.sqrt((2 * math.log(j + 1)) / counts[i])
            upper_bound_all.append(upper_bound)
            lower_bound_all.append(lower_bound)

        selected_machine = np.argmax(upper_bound_all)
        counts[selected_machine] += 1


        avg_reward = sum(rewards) / sum(counts)
        acr.append(avg_reward)
    
    plt.plot(acr,label=f'constant = {constant}')
    constant+=1


plt.legend()
plt.show()