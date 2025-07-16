import matplotlib.pyplot as plt
import numpy as np
import math
import random

def pad_rewards(rewards, target_length):
    return rewards + [float('nan')] * (target_length - len(rewards))

def exponential_decay(machines, spins, BanditMeans):
    epsilon_0 = 1.0
    decay_rate = 0.01
    rewards = [0.0] * machines
    pulls = [0] * machines
    action_regret = []

    optimal_machine = np.argmax(BanditMeans)

    for i in range(spins):
        epsilon = epsilon_0 * math.exp(-decay_rate * i)
        if random.random() > epsilon:
            chosen_machine = np.argmax(rewards)
        else:
            chosen_machine = random.randint(0, machines - 1)

        pulls[chosen_machine] += 1
        regret = i + 1 - pulls[optimal_machine]
        action_regret.append(regret)

    return action_regret

def linear_decay(machines, spins, BanditMeans):
    epsilon_0 = 1.0
    decay_rate = 0.01
    rewards = [0.0] * machines
    pulls = [0] * machines
    action_regret = []

    optimal_machine = np.argmax(BanditMeans)

    for i in range(spins):
        epsilon = epsilon_0 * (decay_rate / (i + 1)) 
        if random.random() > epsilon:
            chosen_machine = np.argmax(rewards)
        else:
            chosen_machine = random.randint(0, machines - 1)

        pulls[chosen_machine] += 1
        regret = i + 1 - pulls[optimal_machine]
        action_regret.append(regret)

    return action_regret

def ucb_method(machines, spins, BanditMeans):
    rewards = [0.0] * machines
    counts = [0] * machines
    action_regret = []

    optimal_machine = np.argmax(BanditMeans)

    for i in range(spins):
        if 0 in counts:
            chosen_machine = counts.index(0)
        else:
            ucb_values = [
                (rewards[j] / counts[j]) + math.sqrt(2 * math.log(i + 1) / counts[j])
                for j in range(machines)
            ]
            chosen_machine = np.argmax(ucb_values)

        counts[chosen_machine] += 1
        regret = i + 1 - counts[optimal_machine]
        action_regret.append(regret)

    return action_regret

def epsilon_greedy(machines, spins, BanditMeans, epsilon=10):
    rewards = [0.0] * machines
    pulls = [0] * machines
    action_regret = []

    optimal_machine = np.argmax(BanditMeans)

    for i in range(spins):
        if i % epsilon != 0:
            chosen_machine = np.argmax(rewards)
        else:
            chosen_machine = random.randint(0, machines - 1)

        pulls[chosen_machine] += 1
        regret = i + 1 - pulls[optimal_machine]
        action_regret.append(regret)

    return action_regret

def softmax_decay(machines, spins, BanditMeans):
    T0 = 1.0
    rewards = [0.0] * machines
    pulls = [0] * machines
    action_regret = []

    optimal_machine = np.argmax(BanditMeans)

    for i in range(spins):
        T = max(T0 * np.exp(-(i**2 / spins)), 1e-10)
        probs = []
        est_reward = []

        max_reward = max(rewards)
        for j in range(machines):
            est_reward.append(np.exp(((rewards[j] - max_reward) / (i + 1)) / T))
        total_est_reward = sum(est_reward)
        probs = [er / total_est_reward for er in est_reward]

        chosen_machine = np.random.choice(machines, p=probs)

        pulls[chosen_machine] += 1
        regret = i + 1 - pulls[optimal_machine]
        action_regret.append(regret)

    return action_regret

def main():
    machines = int(input("Enter the number of machines: "))
    spins = int(input("Enter the number of spins: "))
    repeats = int(input("Enter the number of repetitions for each method: "))
    
    BanditMeans = [np.random.normal(0, 1) for _ in range(machines)]

    methods = {
        '1': ("Exponential Decay", exponential_decay),
        '2': ("Linear Decay", linear_decay),
        '3': ("Upper Bound Confidence (UCB)", ucb_method),
        '4': ("Epsilon-Greedy", epsilon_greedy),
        '5': ("SoftMax with Temperature Decay", softmax_decay)
    }

    print("Select methods to run (separate by commas for multiple):")
    for key, (name, _) in methods.items():
        print(f"{key}: {name}")

    selected_methods = input("Your choice: ").split(',')

    plt.figure(figsize=(12, 8))
    plt.xlabel('Steps')
    plt.ylabel('Action Regret')
    
    for method_key in selected_methods:
        if method_key.strip() in methods:
            method_name, method_func = methods[method_key.strip()]
            cumulative_action_regret = np.zeros(spins)

            for _ in range(repeats):
                action_regret = method_func(machines, spins, BanditMeans)
                cumulative_action_regret += np.array(action_regret)

            avg_action_regret = cumulative_action_regret / repeats
            plt.plot(avg_action_regret, label=f"{method_name} (averaged over {repeats} runs)")

    plt.title("Multi-Armed Bandit Problem - Action Regret Comparison")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
