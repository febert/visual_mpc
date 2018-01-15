import numpy as np

n = 5
sigma = np.concatenate([np.array([4., 4., 1e-5**2]) for _ in range(n)])
sigma = np.diag(sigma)
mean = np.zeros(sigma.shape[0])

smp = np.random.multivariate_normal(mean, sigma)
print smp.reshape([n,3])