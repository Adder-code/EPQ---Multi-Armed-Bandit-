import matplotlib.pyplot as plt
import numpy as np
import math

def ucb_method(machines, spins, BanditMeans):
    # Initialize variables
    rewards = [0.0] * machines  # Total rewards for each machine
    counts = [0] * machines     # Number of times each machine is pulled
    acr = []                    # List to store cumulative average rewards
    
    # Pull each machine once to initialize
    for i in range(machines):
        reward = np.random.normal(BanditMeans[i], 1)
        rewards[i] += reward
        counts[i] += 1

    # UCB formula: avg_reward + sqrt(2 * log(n) / counts)
    for j in range(machines, spins):
        upper_bound_all = []
        for i in range(machines):
            avg_reward = rewards[i] / counts[i]
            exploration_term = math.sqrt((2 * math.log(j + 1)) / counts[i])
            upper_bound = avg_reward + exploration_term
            upper_bound_all.append(upper_bound)

        # Select the machine with the highest UCB
        selected_machine = np.argmax(upper_bound_all)
        reward = np.random.normal(BanditMeans[selected_machine], 1)

        # Update rewards and counts
        rewards[selected_machine] += reward
        counts[selected_machine] += 1

        # Update cumulative average reward
        cumulative_avg_reward = sum(rewards) / sum(counts)
        acr.append(cumulative_avg_reward)
    
    return acr

def main():
    machines = int(input("Enter the number of machines: "))
    spins = int(input("Enter the number of spins: "))
    repeats = int(input("Enter the number of repetitions for UCB: "))

    # Generate random mean rewards for each machine
    BanditMeans = [np.random.normal(0, 1) for _ in range(machines)]

    plt.figure(figsize=(12, 8))
    plt.xlabel("Steps")
    plt.ylabel("Cumulative Average Reward")
    plt.title("Upper Bound Confidence (UCB)")

    cumulative_rewards = np.zeros(spins)

    for _ in range(repeats):
        rewards = ucb_method(machines, spins, BanditMeans)
        cumulative_rewards += rewards
        plt.plot(rewards, color="lightgray", linewidth=0.8, alpha=0.7)

    cumulative_rewards /= repeats
    plt.plot(cumulative_rewards, label="UCB (averaged)", linewidth=2, color="blue")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
