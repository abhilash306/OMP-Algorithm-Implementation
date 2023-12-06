import numpy as np
import matplotlib.pyplot as plt

# Provided values for M and N
M_values = [200, 400, 600]
N_values = [1000, 4000]

# Plotting the sensing matrices Φ for different values of M and N as grey level matrices
plt.figure(figsize=(18, 6))

for i, N in enumerate(N_values):
    for j, M in enumerate(M_values):
        # Create the sensing matrix
        Phi = np.random.normal(0, 1, (M, N))

        # Plotting
        plt.subplot(len(N_values), len(M_values), i * len(M_values) + j + 1)
        plt.imshow(Phi, cmap='gray', aspect='auto')
        plt.colorbar()
        plt.title(f'Sensing Matrix Φ (M={M}, N={N})')
        plt.xlabel('N')
        plt.ylabel('M')

plt.tight_layout()

# Saving the figure to a file
plt.savefig('sensing_matrices.png')

# Show the plot
plt.show()
