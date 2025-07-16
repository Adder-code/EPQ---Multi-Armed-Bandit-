import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(20, 16))
plt.xlabel('Steps')
plt.ylabel('Average Reward')

variance = int(input('variance - '))
machines = int(input('machines - '))
spins = int(input('spins - '))


BanditMeans = [np.random.normal(0, 1) for _ in range(machines)]


all_averages = []

for j in range(machines):
    banditr = []  

    averages = []  

    for i in range(spins):
        reward = np.random.normal(BanditMeans[j], variance)
        banditr.append(reward)
        avg = sum(banditr) / len(banditr)
        averages.append(avg)

    all_averages.append(averages)

all_avg = np.zeros(spins)

for j in range(machines):
    avg_flt = list(map(float, all_averages[j]))
    '''if all(x > 0 for x in avg_flt):
        plt.plot(avg_flt, label=f'Machine {j + 1}')'''
    plt.plot(avg_flt)
    all_avg+=avg_flt

all_avg/=machines

plt.plot(all_avg, color='black', label='all average')

plt.legend()
plt.show()


