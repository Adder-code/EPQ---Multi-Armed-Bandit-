
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

    for i in range(machines):
        reward = np.random.normal(BanditMeans[i], 1)
        rewards[i] += reward
        counts[i] += 1

    for j in range(machines, spins):
        upper_bound_all = []
        for i in range(machines):
            avg_reward = rewards[i] / counts[i]
            exploration_term = constant * math.sqrt((2 * math.log(j + 1)) / counts[i])
            upper_bound = avg_reward + exploration_term
            upper_bound_all.append(upper_bound)
        
        selected_machine = np.argmax(upper_bound_all)
        reward = np.random.normal(BanditMeans[selected_machine], 1)
        rewards[selected_machine] += reward
        counts[selected_machine] += 1

        avg_reward = sum(rewards) / sum(counts)
        acr.append(avg_reward)
    
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

def softmax_decay(machines, spins, BanditMeans):
    T0 = 1.0
    rewards = [0.0] * machines
    pulls = [0] * machines
    acr = []
    cumulative_reward = 0.0

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

    # Initialize each machine once (ensuring UCB starts with valid values)
    for i in range(machines):
        reward = np.random.normal(BanditMeans[i], 1)
        rewards[i] += reward
        counts[i] += 1
        cumulative_reward += reward

    # Main loop for action selection
    for i in range(machines, spins):
        epsilon = epsilon_0 * math.exp(-decay_rate * i)  # Exponential decay exploration rate

        if random.random() < epsilon:
            # Exploration: Choose a random machine
            selected_machine = random.randint(0, machines - 1)
        else:
            # Exploitation: Use UCB without knowing total spins
            upper_bound_all = []
            for j in range(machines):
                avg_reward = rewards[j] / counts[j]
                exploration_term = math.sqrt((2 * math.log(sum(counts))) / counts[j])  # Uses past observations only
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
        '5': ("SoftMax with Temperature Decay", softmax_decay),
        '6': ("Hybrid Exponential-UCB", hybrid_exponential_ucb)
    }

    print("Select methods to run (separate by commas for multiple):")
    for key, (name, _) in methods.items():
        print(f"{key}: {name}")

    selected_methods = input("Your choice: ").split(',')

    plt.figure(figsize=(12, 8))
    plt.xlabel('Steps')
    plt.ylabel('Regret')

    for method_key in selected_methods:
        if method_key.strip() in methods:
            method_name, method_func = methods[method_key.strip()]
            all_runs = [] 
            cumulative_rewards = np.zeros(spins)

            for _ in range(repeats):
                rewards = method_func(machines, spins, BanditMeans)
                rewards = pad_rewards(rewards, spins)  
                all_runs.append(rewards)
                cumulative_rewards += np.nan_to_num(rewards)
                regret = []
                regret.append(rewards[0])
                for i in range(1, spins):
                    regret.append((max(BanditMeans) - rewards[i]) + regret[i - 1])
                plt.plot(regret, color='lightgray', linewidth=0.8, alpha=0.7)

            cumulative_rewards /= repeats

            cumulative_rewards = np.where(cumulative_rewards == 0, float('nan'), cumulative_rewards)
            avg_regret = []
            avg_regret.append(cumulative_rewards[0])
            for i in range(1, spins):
                avg_regret.append((max(BanditMeans) - cumulative_rewards[i]) + avg_regret[i - 1])
            plt.plot(avg_regret, label=f"{method_name} (averaged over {repeats} runs)", linewidth=2)

    plt.title("Multi-Armed Bandit Problem - Comparison of Methods")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
