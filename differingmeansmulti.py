from combinedmulti import exponential_decay, linear_decay, ucb_method, epsilon_greedy, softmax_decay
import numpy as np
import matplotlib.pyplot as plt


def capped_normal(mean, std_dev, max_value):
    while True:
        value = np.random.normal(mean, std_dev)
        if value <= max_value:
            return value


def generate_bandit_means(machines, mean, max_mean):
    means = [capped_normal(mean, 1, max_mean) for _ in range(machines - 1)]
    means.append(max_mean)
    np.random.shuffle(means)
    return means


def repeat_experiments_varying_means(machines, spins, repeats, initial_max_mean, step, iterations, methods):
    results = {}
    for method_name, method_func in methods.items():
        results[method_name] = []
        for i in range(iterations):
            current_mean = initial_max_mean + i * step
            for _ in range(repeats):
                bandit_means = generate_bandit_means(machines, current_mean, current_mean + 1)
                rewards = method_func(machines, spins, bandit_means)
                results[method_name].append((current_mean, rewards))
    return results


if __name__ == "__main__":
    machines = int(input("Enter the number of machines: "))
    spins = int(input("Enter the number of spins: "))
    repeats = int(input("Enter the number of repetitions: "))
    initial_max_mean = float(input("Enter the initial mean value: "))
    step = float(input("Enter the increment step for maximum mean: "))
    iterations = int(input("Enter the number of steps: "))

    available_methods = {
        "Exponential Decay": exponential_decay,
        "Linear Decay": linear_decay,
        "Upper Bound Confidence (UCB)": ucb_method,
        "Epsilon-Greedy": epsilon_greedy,
        "SoftMax Decay": softmax_decay,
    }

    method_colors = {
        "Exponential Decay": "blue",
        "Linear Decay": "green",
        "Upper Bound Confidence (UCB)": "orange",
        "Epsilon-Greedy": "red",
        "SoftMax Decay": "purple",
    }

    print("Available methods:")
    for i, method_name in enumerate(available_methods.keys(), 1):
        print(f"{i}: {method_name}")
    selected_methods = input("Enter the numbers of the methods you want to use (comma-separated): ")
    selected_methods = [list(available_methods.keys())[int(i) - 1] for i in selected_methods.split(",")]

    selected_methods_dict = {name: available_methods[name] for name in selected_methods}

    results = repeat_experiments_varying_means(
        machines, spins, repeats, initial_max_mean, step, iterations, selected_methods_dict
    )

    plt.figure(figsize=(12, 8))
    plt.xlabel("(Mean of Machine Distribution)")
    plt.ylabel("Cumulative Average Reward")

    for method_name, method_results in results.items():
        grouped_rewards = {}
        for current_mean, rewards in method_results:
            if current_mean not in grouped_rewards:
                grouped_rewards[current_mean] = []
            grouped_rewards[current_mean].append(np.mean(rewards))

        x = sorted(grouped_rewards.keys())
        y = [np.mean(grouped_rewards[mean]) for mean in x]
        plt.plot(x, y, label=method_name, color=method_colors[method_name])

    plt.title("Comparison of Multi-Armed Bandit Strategies with Incremented Machine Means")
    plt.legend()
    plt.show()
