"""Gaussian KBLUP, one model per module. Bandwidth is learned from train only."""
import numpy as np
from .preprocessing import normalize_kernel, kernel_eigh
from .gblup import gblup_fit_reml, gblup_predict


def gaussian_kernel(A, B, bandwidth):
    A, B = np.asarray(A, float), np.asarray(B, float)
    squared = np.maximum(np.sum(A*A, axis=1)[:, None]
                         + np.sum(B*B, axis=1)[None, :] - 2*A@B.T, 0.)
    return np.exp(-squared / bandwidth)


def prepare_kblup(X_train, X_test, F=None):
    X_train, X_test = np.asarray(X_train, float), np.asarray(X_test, float)
    keep = X_train.var(axis=0) > 0
    if not keep.any():
        raise ValueError("No polymorphic training markers.")
    Z, T = X_train[:, keep], X_test[:, keep]
    norm = np.sum(Z*Z, axis=1)
    distances = np.maximum(norm[:, None]+norm[None, :]-2*Z@Z.T, 0.)
    positive = distances[np.triu_indices(len(Z), 1)]
    positive = positive[positive > 0]
    if not len(positive):
        raise ValueError("All training profiles are identical.")
    bandwidth = float(np.median(positive))
    K = np.exp(-distances / bandwidth)
    cross = gaussian_kernel(T, Z, bandwidth)
    K, cross, scale = normalize_kernel(K, cross, F)
    return dict(K=K, cross=cross, eigen=kernel_eigh(K), scale=scale,
                bandwidth=bandwidth, keep=keep, n_markers=int(keep.sum()))


def kblup_fit(y, prepared, F=None, bounds=(-12., 12.)):
    fit = gblup_fit_reml(y, prepared["K"], F=F, eigen=prepared["eigen"], bounds=bounds)
    fit["bandwidth"] = prepared["bandwidth"]
    return fit


def kblup_predict(fit, prepared, F_new=None):
    return gblup_predict(fit, prepared["cross"], F_new)
