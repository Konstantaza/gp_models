"""GBLUP — тот же BLUP, но через матрицу родства вместо маркеров.

Модель:  y = 1 mu + g + e,  g ~ N(0, G s2g),  e ~ N(0, I s2e).

Решение (Хендерсон):
    V  = s2g * G + s2e * I
    mu = (1' V^-1 1)^-1 1' V^-1 y
    g  = s2g * G V^-1 (y - 1 mu)

При G = ZZ'/c и s2g = c * s2_beta получается g = Z beta из RR-BLUP —
одна модель в двух записях (численно проверено в ноутбуке 02).
"""

import numpy as np

from .preprocessing import reml_variance_components


def gblup_fit(y, K, s2g, s2e):
    """GBLUP с заданными дисперсиями. K — матрица родства (или любое ядро)."""
    y = np.asarray(y, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    n = y.shape[0]
    one = np.ones(n)
    V = s2g * K + s2e * np.eye(n)
    Vi = np.linalg.inv(V)
    mu = (one @ Vi @ y) / (one @ Vi @ one)
    g = s2g * K @ Vi @ (y - one * mu)
    return dict(g=g, mu=mu, gebv=mu + g)


def gblup_fit_reml(y, K):
    """Сначала REML-оценка дисперсий, потом GBLUP. Отдаёт dict(g, mu, gebv, vc)."""
    vc = reml_variance_components(y, K)
    fit = gblup_fit(y, K, vc["sigma2_g"], vc["sigma2_e"])
    fit["vc"] = vc
    return fit


def gblup_fit_predict(y_tr, K_trtr, K_tetr, s2g, s2e):
    """Учит на train и прогнозирует test через их родство с train:
    g_te = Cov(g_te, g_tr') V_tr^-1 (y_tr - 1 mu) = s2g * K_te,tr V^-1 (y_tr - 1 mu).
    Отдаёт (pred, fit)."""
    y_tr = np.asarray(y_tr, dtype=np.float64)
    fit = gblup_fit(y_tr, K_trtr, s2g, s2e)
    n = y_tr.shape[0]
    V = s2g * K_trtr + s2e * np.eye(n)
    Vi = np.linalg.inv(V)
    pred = fit["mu"] + s2g * np.asarray(K_tetr) @ Vi @ (y_tr - fit["mu"])
    return pred, fit
