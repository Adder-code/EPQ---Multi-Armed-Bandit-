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
    acr = []
    cumulative_reward = 0.0

    for i in range(spins):
        epsilon = epsilon_0 * math.exp(-decay_rate * i)
        if random.random() > epsilon:
            selected_machine = np.argmax(rewards)
        else:
            selected_machine = random.randint(0, len(rewards) - 1)

        reward = np.random.normal(BanditMeans[selected_machine], 1)
        rewards[selected_machine] += reward
        pulls[selected_machine] += 1
        cumulative_reward += reward
        cumulative_avg_reward = cumulative_reward / (i + 1)
        acr.append(cumulative_avg_reward)
    
    return acr

def linear_decay(machines, spins, BanditMeans):
    epsilon_0 = 1.0
    decay_rate = 0.01
    rewards = [0.0] * machines
    pulls = [0] * machines
    acr = []
    cumulative_reward = 0.0

    for i in range(spins):
        epsilon = epsilon_0 * (decay_rate / (i + 1))
        if random.random() > epsilon:
            chosen_machine = np.argmax(rewards)
        else:
            chosen_machine = random.randint(0, len(rewards) - 1)

        reward = np.random.normal(BanditMeans[chosen_machine], 1)
        rewards[chosen_machine] += reward
        pulls[chosen_machine] += 1
        cumulative_reward += reward
        cumulative_avg_reward = cumulative_reward / (i + 1)
        acr.append(cumulative_avg_reward)
    
    return acr

def ucb_method(machines, spins, BanditMeans):
    constant = math.sqrt(2 * math.log(spins) / machines)
    rewards = [0.0] * machines
    counts = [0] * machines
    acr = []

    # Initial pulls for each machine (recording cumulative average)
    for i in range(machines):
        reward = np.random.normal(BanditMeans[i], 1)
        rewards[i] += reward
        counts[i] += 1
        current_avg = sum(rewards) / sum(counts)
        acr.append(current_avg)

    # UCB action selection for remaining spins
    for j in range(machines, spins):
        upper_bound_all = []
        for i in range(machines):
            avg_reward_machine = rewards[i] / counts[i]
            exploration_term = constant * math.sqrt((2 * math.log(j + 1)) / counts[i])
            upper_bound = avg_reward_machine + exploration_term
            upper_bound_all.append(upper_bound)
        
        selected_machine = np.argmax(upper_bound_all)
        reward = np.random.normal(BanditMeans[selected_machine], 1)
        rewards[selected_machine] += reward
        counts[selected_machine] += 1
        current_avg = sum(rewards) / sum(counts)
        acr.append(current_avg)
    
    return acr

def epsilon_greedy(machines, spins, BanditMeans, epsilon=10):
    rewards = [0.0] * machines
    pulls = [0] * machines
    acr = []
    cumulative_reward = 0.0

    for i in range(spins):
        if i % epsilon != 0:
            chosen_machine = np.argmax(rewards)
        else:
            chosen_machine = random.randint(0, len(rewards) - 1)

        reward = np.random.normal(BanditMeans[chosen_machine], 1)
        rewards[chosen_machine] += reward
        pulls[chosen_machine] += 1
        cumulative_reward += reward
        cumulative_avg_reward = cumulative_reward / (i + 1)
        acr.append(cumulative_avg_reward)
    
    return acr

def softmax_decay(machines, spins, BanditMeans, T0=1.0):
    rewards = [0.0] * machines
    pulls = [0] * machines
    acr = []
    cumulative_reward = 0.0

    for i in range(spins):
        T = T0 * np.exp(-i / spins)
        avg_rewards = [rewards[j] / pulls[j] if pulls[j] > 0 else 0 for j in range(machines)]
        exp_values = np.exp(np.array(avg_rewards) / T)
        probabilities = exp_values / np.sum(exp_values)
        chosen_machine = np.random.choice(machines, p=probabilities)
        reward = np.random.normal(BanditMeans[chosen_machine], 1)
        rewards[chosen_machine] += reward
        pulls[chosen_machine] += 1
        cumulative_reward += reward
        cumulative_avg_reward = cumulative_reward / (i + 1)
        acr.append(cumulative_avg_reward)
    
    return acr

def hybrid_exponential_ucb(machines, spins, BanditMeans):
    epsilon_0 = 1.0  # Initial exploration rate
    decay_rate = 0.01  # Exponential decay rate
    rewards = [0.0] * machines
    counts = [0] * machines
    acr = []
    cumulative_reward = 0.0

    # Initial pulls for each machine (recording cumulative average)
    for i in range(machines):
        reward = np.random.normal(BanditMeans[i], 1)
        rewards[i] += reward
        counts[i] += 1
        cumulative_reward += reward
        acr.append(cumulative_reward / (i + 1))

    # Main loop for action selection
    for i in range(machines, spins):
        epsilon = epsilon_0 * math.exp(-decay_rate * i)
        if random.random() < epsilon:
            selected_machine = random.randint(0, machines - 1)
        else:
            upper_bound_all = []
            for j in range(machines):
                avg_reward = rewards[j] / counts[j]
                exploration_term = math.sqrt((2 * math.log(sum(counts))) / counts[j])
                upper_bound = avg_reward + exploration_term
                upper_bound_all.append(upper_bound)
            selected_machine = np.argmax(upper_bound_all)
        reward = np.random.normal(BanditMeans[selected_machine], 1)
        rewards[selected_machine] += reward
        counts[selected_machine] += 1
        cumulative_reward += reward
        cumulative_avg_reward = cumulative_reward / (i + 1)
        acr.append(cumulative_avg_reward)
    return acr

def main():
    machines = int(input("Enter the number of machines: "))
    spins = int(input("Enter the number of spins: "))
    repeats = int(input("Enter the number of repetitions: "))
    variance = 1
    BanditMeans = [np.random.normal(0, variance) for _ in range(machines)]
    methods = {
        '1': ("Exponential Decay", exponential_decay),
        '2': ("Linear Decay", linear_decay),
        '3': ("Upper Bound Confidence (UCB)", ucb_method),
        '4': ("Epsilon-Greedy", epsilon_greedy),
        '5': ("SoftMax with Temperature Decay", softmax_decay),
        '6': ("Hybrid UCB + Exponential Decay", hybrid_exponential_ucb)
    }

    print("Select methods to run (separate by commas for multiple):")
    for key, (name, _) in methods.items():
        print(f"{key}: {name}")

    selected_methods = input("Your choice: ").split(',')
    plt.figure(figsize=(12, 8))
    plt.xlabel('Steps')
    plt.ylabel('Cumulative Average Reward')
    
    method_results = {}

    for method_key in selected_methods:
        method_key = method_key.strip()
        if method_key in methods:
            method_name, method_func = methods[method_key]
            cumulative_rewards = np.zeros(spins)
            for _ in range(repeats):
                rewards = method_func(machines, spins, BanditMeans)
                rewards = pad_rewards(rewards, spins)
                cumulative_rewards += np.nan_to_num(rewards)
                plt.plot(rewards, color='lightgray', linewidth=0.8, alpha=0.7)
            cumulative_rewards /= repeats
            cumulative_rewards = np.where(cumulative_rewards == 0, float('nan'), cumulative_rewards)
            method_results[method_name] = cumulative_rewards
            plt.plot(cumulative_rewards, label=f"{method_name} (averaged over {repeats} runs)", linewidth=2)
    
    best_machine_index = np.argmax(BanditMeans)
    Best_means_avg = []
    cumulative_best_reward = 0.0
    for k in range(spins):
        reward = np.random.normal(BanditMeans[best_machine_index], 1)
        cumulative_best_reward += reward
        Best_means_avg.append(cumulative_best_reward / (k + 1))
    
    for method_name, cumulative_rewards in method_results.items():
        final_optimal = Best_means_avg[-1]
        final_method = cumulative_rewards[-1]
        final_regret = final_optimal - final_method
        percentage_regret = (final_regret / final_optimal) * 100 if final_optimal != 0 else float('nan')
        print(f'{method_name} -> Numerical Regret: {final_regret:.4f}, Percentage Regret: {percentage_regret:.2f}%')

    plt.plot(Best_means_avg, label="Optimal", color='blue', linestyle='--')
    plt.title("Multi-Armed Bandit Problem - Comparison of Methods")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
