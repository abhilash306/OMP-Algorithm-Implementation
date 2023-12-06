import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
from scipy.linalg import lstsq

def omp(phi, y, e):
    n, m = phi.shape
    Xhat = np.zeros(m)
    rhat = y
    lambda_set = []

    while np.linalg.norm(rhat) > e:
        product = np.dot(phi.T, rhat)
        location = np.argmax(np.abs(product))
        lambda_set.append(location)

        phi_k = phi[:, lambda_set]
        x_k, _, _, _ = lstsq(phi_k, y)
        Xhat[lambda_set] = x_k

        yhat = np.dot(phi, Xhat)
        rhat = y - yhat

    return Xhat

# Inputs from user
n = int(input('Please enter value of signal length(n): '))
p = float(input('Please enter value of probability(p): '))
e = float(input('Please enter value of threshold(e): '))

# Input signal generator
A = np.random.rand(n) < p
X = np.random.randn(n) * A

# Sparsity indicator
h = np.arange(n+1)
q = binom.pmf(h, n, p)

plt.figure()
plt.bar(h, q, 1)
plt.xlabel("Sparsity level")
plt.ylabel("Probability")
plt.title(f"N = {n} p = {p}")
plt.show()

errors = []
K_hat = []

for m in [200, 400, 600]:
    K = np.sum(A)
    phi = np.random.randn(m, n)
    y = np.dot(phi, X)

    # OMP recovery
    Xhat = omp(phi, y, e)

    # Comparison plot for original and recovered signal
    plt.figure()
    plt.stem(X, 'r', markerfmt='ro', label='X')
    plt.stem(Xhat, 'b', markerfmt='bo', label='Xhat')
    plt.legend()
    plt.title(f"N = {n} M = {m} p = {p}")
    plt.show()

    errors.append(np.linalg.norm(X - Xhat))
    K_hat.append(np.count_nonzero(Xhat))

# Display values
print("Error:", errors)
print("K_hat:", K_hat)
print("K:", K)
