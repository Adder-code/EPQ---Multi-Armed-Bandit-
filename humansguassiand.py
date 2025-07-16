import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Mean and standard deviation for male height (in cm)
mu = 175
sigma = 7

# Generate height data
heights = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
pdf = stats.norm.pdf(heights, mu, sigma)

# Plot the normal distribution
plt.figure(figsize=(8, 5))
plt.plot(heights, pdf, label="Normal Distribution of Heights", color="blue")
plt.axvline(mu, color='red', linestyle="dashed", label="Mean Height (175 cm)")
plt.axvline(mu + sigma, color='green', linestyle="dashed", label="1 SD Above (182 cm)")
plt.axvline(mu - sigma, color='green', linestyle="dashed", label="1 SD Below (168 cm)")
plt.fill_between(heights, pdf, where=((heights >= 168) & (heights <= 182)), color='green', alpha=0.3)

plt.title("Height Distribution of Adult Males")
plt.xlabel("Height (cm)")
plt.ylabel("Probability Density")
plt.legend()
plt.show()
