import matplotlib.pyplot as plt
import numpy as np

# Parameters
mu_x, mu_z = 0, 0  # Means
sigma = 1  # Standard deviation
extent = 5  # Range for x and z

# Create a grid of x and z values
x = np.linspace(-extent, extent, 500)
z = np.linspace(-extent, extent, 500)
X, Z = np.meshgrid(x, z)

# Calculate the 3D Gaussian function
Y = (1 / (2 * np.pi * sigma**2)) * np.exp(-((X - mu_x)**2 + (Z - mu_z)**2) / (2 * sigma**2))

# Plotting the 3D surface
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Z, Y, cmap='viridis', edgecolor='none', alpha=0.8)

# Add lines for the 68-95-99.7 rule on x and z axes
for i in range(1, 4):  # 1, 2, and 3 standard deviations
    # Lines for the x-axis
    x_line = np.full_like(z, mu_x - i * sigma)  # Create an array with the same shape as z
    z_line = z  # Use the existing z array
    y_line = (1 / (2 * np.pi * sigma**2)) * np.exp(-((x_line - mu_x)**2 + (z_line - mu_z)**2) / (2 * sigma**2))
    ax.plot(x_line, z_line, y_line, color='red', linestyle='dotted', linewidth=1.5, label=f'{i}σ' if i == 1 else None)

    x_line = np.full_like(z, mu_x + i * sigma)  # Create an array with the same shape as z
    y_line = (1 / (2 * np.pi * sigma**2)) * np.exp(-((x_line - mu_x)**2 + (z_line - mu_z)**2) / (2 * sigma**2))
    ax.plot(x_line, z_line, y_line, color='red', linestyle='dotted', linewidth=1.5)

    # Lines for the z-axis
    z_line = np.full_like(x, mu_z - i * sigma)  # Create an array with the same shape as x
    x_line = x  # Use the existing x array
    y_line = (1 / (2 * np.pi * sigma**2)) * np.exp(-((x_line - mu_x)**2 + (z_line - mu_z)**2) / (2 * sigma**2))
    ax.plot(x_line, z_line, y_line, color='blue', linestyle='dotted', linewidth=1.5, label=f'{i}σ' if i == 1 else None)

    z_line = np.full_like(x, mu_z + i * sigma)  # Create an array with the same shape as x
    y_line = (1 / (2 * np.pi * sigma**2)) * np.exp(-((x_line - mu_x)**2 + (z_line - mu_z)**2) / (2 * sigma**2))
    ax.plot(x_line, z_line, y_line, color='blue', linestyle='dotted', linewidth=1.5)



ax.set_title("3D Normal Distribution with 68-95-99.7 Rule")
ax.set_xlabel("X (Input Variable)")
ax.set_ylabel("Z (Input Variable)")
ax.set_zlabel("Probability Density (Y)")


ax.legend(["1σ = 68%", "2σ = 95%", "3σ = 99.7%"], loc="upper right")


fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

plt.show()
