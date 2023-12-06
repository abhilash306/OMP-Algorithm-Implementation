import numpy as np
import matplotlib.pyplot as plt

# Parameters
p = 0.2  # Probability of a non-zero element
N = 1000  # Length of the sequence
num_realizations = 1000  # Number of realizations for the ensemble

# Generate realizations
sparsity_counts = []

for _ in range(num_realizations):
    # Bernoulli sequence for support
    support = np.random.choice([0, 1], size=N, p=[1-p, p])

    # Gaussian random variables for non-zero amplitudes
    amplitudes = np.random.normal(0, 1, size=N)
    signal = support * amplitudes

    # Count non-zero elements (sparsity)
    sparsity_counts.append(np.count_nonzero(signal))

# Calculate average sparsity
average_sparsity = np.mean(sparsity_counts)

# Plotting the histogram of sparsity counts
plt.hist(sparsity_counts, bins=30, color='blue', alpha=0.7)
plt.title('Histogram of Sparsity Counts')
plt.xlabel('Number of Non-Zero Elements')
plt.ylabel('Frequency')
plt.axvline(average_sparsity, color='red', linestyle='dashed', linewidth=2)
plt.text(average_sparsity + 5, max(plt.ylim())*0.9, f'Average: {average_sparsity:.2f}', color='red')

# Saving the plot as an image
plt.savefig('sparsity_histogram.png')

# Show the plot
plt.show()
