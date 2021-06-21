import time
import numpy as np
from numpy.linalg import norm

from andersoncd.weighted_lasso import celer_primal
from celer.datasets import make_correlated_data

X, y, _ = make_correlated_data(
    n_samples=1000, n_features=5_000, snr=5, corr=0.7, random_state=1,
    density=0.2)
n_samples, n_features = X.shape
alpha_max = norm(X.T @ y, ord=np.inf) / n_samples
alpha = 0.01 * alpha_max


w = np.zeros(n_features)
R = y - X @ w
norms_X_col = norm(X, axis=0)
weights = np.ones(n_features)

t_start = time.time()
celer_primal(
    X, y, alpha, w, R, norms_X_col, weights, max_iter=10,
    max_epochs=1_000, p0=10, use_acc=True, K=5, verbose=True,
    tol=1e-10)
t_ellapsed = time.time() - t_start
print(t_ellapsed)
