"""Gaussian KBLUP: импутация, маска и bandwidth только по train."""

import numpy as np
from .preprocessing import prepare_markers, normalize_kernel, kernel_eigh
from .gblup import gblup_fit_reml, gblup_predict


def gaussian_kernel(A, B, bandwidth):
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("bandwidth must be positive and finite.")
    A, B = np.asarray(A, float), np.asarray(B, float)
    squared = np.maximum(np.sum(A*A, axis=1)[:, None]
                         + np.sum(B*B, axis=1)[None, :] - 2*A@B.T, 0.)
    return np.exp(-squared / bandwidth)


def prepare_kblup(X_train, X_test, F=None, coding="dosage"):
    prep = prepare_markers(X_train, X_test, coding)
    A, B = prep["A"], prep["B"]
    norm = np.sum(A*A, axis=1)
    distances = np.maximum(norm[:, None] + norm[None, :] - 2*A@A.T, 0.)
    positive = distances[np.triu_indices(len(A), 1)]
    positive = positive[positive > 1e-12]
    if not len(positive):
        raise ValueError("All training profiles are identical.")
    bandwidth = float(np.median(positive))
    K = np.exp(-distances / bandwidth)
    cross = gaussian_kernel(B, A, bandwidth)
    K, cross, scale = normalize_kernel(K, cross, F)
    return dict(K=K, cross=cross, eigen=kernel_eigh(K), scale=scale,
                bandwidth=bandwidth, **prep)


def kblup_fit(y, prepared, F=None, bounds=(-8., 8.)):
    fit = gblup_fit_reml(y, prepared["K"], F=F,
                         eigen=prepared["eigen"], bounds=bounds)
    fit["bandwidth"] = prepared["bandwidth"]
    return fit


def kblup_predict(fit, prepared, F_new=None):
    return gblup_predict(fit, prepared["cross"], F_new)
