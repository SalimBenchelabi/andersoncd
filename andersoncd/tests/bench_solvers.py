import time
import numpy as np

from andersoncd import WeightedLasso
from celer.datasets import make_correlated_data

X, y, _ = make_correlated_data(
    n_samples=1000, n_features=5_000, snr=5, corr=0.7, random_state=1,
    density=0.2)


t_start = time.time()
weights = np.random.rand(X.shape[1])
weights[:50] = 0
alpha_max = np.max(np.abs(X[:, 50:].T @ y)) / len(y)
clf = WeightedLasso(weights=weights, alpha=alpha_max/50, verbose=2)
clf.fit(X, y)
t_elapsed = time.time() - t_start
print(f'Time taken: {t_elapsed:.3f} s')
