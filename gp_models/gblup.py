"""GBLUP: старые вызовы сохранены, прогноз через solve, а не inverse."""

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from .preprocessing import (reml_variance_components, prepare_markers,
                            fixed_design, kernel_eigh)


def prepare_gblup(X_train, X_test, F=None, coding="dosage"):
    """Train-only VanRaden G и test×train блок; F проверяется при fit."""
    prep = prepare_markers(X_train, X_test, coding)
    K = prep["Z"] @ prep["Z"].T / prep["c"]
    cross = prep["T"] @ prep["Z"].T / prep["c"]
    return dict(K=K, cross=cross, eigen=kernel_eigh(K), **prep)


def gblup_fit(y, K, s2g, s2e, F=None):
    """Заданные компоненты дисперсии; F по умолчанию intercept."""
    y, K = np.asarray(y, float), np.asarray(K, float)
    n = len(y)
    if y.ndim != 1 or not np.isfinite(y).all() or K.shape != (n, n):
        raise ValueError("Finite 1D y and matching K are required.")
    if not np.isfinite(K).all() or not np.allclose(K, K.T):
        raise ValueError("K must be finite and symmetric.")
    if not np.isfinite([s2g, s2e]).all() or s2g < 0 or s2e <= 0:
        raise ValueError("Require sigma2_g >= 0 and sigma2_e > 0.")
    F = fixed_design(n, F)
    chol = cho_factor(s2g * K + s2e * np.eye(n), check_finite=False)
    ViF = cho_solve(chol, F, check_finite=False)
    Viy = cho_solve(chol, y, check_finite=False)
    beta = np.linalg.solve(F.T @ ViF, F.T @ Viy)
    alpha = s2g * cho_solve(chol, y - F @ beta, check_finite=False)
    g = K @ alpha
    return dict(g=g, mu=float(beta[0]), gebv=F @ beta + g,
                beta_fixed=beta, alpha=alpha, n_train=n,
                intercept_only=F.shape[1] == 1 and np.allclose(F, 1.))


def gblup_fit_reml(y, K, F=None, eigen=None, bounds=(-8., 8.)):
    """REML + fit; прежний gblup_fit_reml(y, K) работает без изменений."""
    vc = reml_variance_components(y, K, bounds=bounds, F=F, eigen=eigen)
    fit = gblup_fit(y, K, vc["sigma2_g"], vc["sigma2_e"], F=F)
    fit["vc"] = vc
    return fit


def gblup_predict(fit, K_tetr, F_new=None):
    """Прогноз новых наблюдений; порядок train-столбцов должен совпадать."""
    cross = np.asarray(K_tetr, float)
    if (cross.ndim != 2 or cross.shape[1] != fit["n_train"]
            or not np.isfinite(cross).all()):
        raise ValueError("cross must be finite test×train in training order.")
    if F_new is None:
        if not fit["intercept_only"]:
            raise ValueError("Provide F_new with the training fixed-effect columns.")
        F_new = np.ones((len(cross), 1))
    F_new = np.asarray(F_new, float)
    if F_new.ndim == 1:
        F_new = F_new[:, None]
    if F_new.shape != (len(cross), len(fit["beta_fixed"])) or not np.isfinite(F_new).all():
        raise ValueError("F_new has incorrect dimensions/values.")
    return F_new @ fit["beta_fixed"] + cross @ fit["alpha"]


def gblup_fit_predict(y_tr, K_trtr, K_tetr, s2g, s2e):
    """Старый интерфейс: возвращает (prediction, fit)."""
    fit = gblup_fit(y_tr, K_trtr, s2g, s2e)
    return gblup_predict(fit, K_tetr), fit
