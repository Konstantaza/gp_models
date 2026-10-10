"""RR-BLUP — ridge-регрессия на маркерах, замкнутая форма.

Модель:  y = 1 mu + Z beta + e,
         beta_j ~ N(0, s2_beta) i.i.d.,  e ~ N(0, I s2_e).

Смешанные уравнения Хендерсона (mu поглощается центрированием Z):
    (Z'Z + lam I) beta = Z'(y - mean(y)),   lam = s2_e / s2_beta.

Это в точности ridge с параметром lam — отсюда название.
Связь с GBLUP: при G = ZZ'/c и s2_g = c * s2_beta геномные значения
g = Z beta совпадают с GBLUP-решением (проверено в ноутбуке 02).
"""

import numpy as np

from .preprocessing import center_markers, reml_variance_components


def rrblup_fit(y, Z, lam):
    """Решает (Z'Z + lam I) beta = Z'(y - mean(y)).

    y — фенотипы train, Z — центрированные маркеры (n x p),
    lam = s2_e / s2_beta. Отдаёт dict(beta, mu, gebv).
    """
    y = np.asarray(y, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    mu = y.mean()
    yc = y - mu
    p = Z.shape[1]
    A = Z.T @ Z + lam * np.eye(p)
    beta = np.linalg.solve(A, Z.T @ yc)   # solve, а не inv: так точнее и быстрее
    return dict(beta=beta, mu=mu, gebv=mu + Z @ beta)


def predict_gebv(fit, Z_new):
    """GEBV для новых линий: g = mu + Z_new beta."""
    return fit["mu"] + np.asarray(Z_new, dtype=np.float64) @ fit["beta"]


def lam_from_h2(X, h2):
    """lambda из заданной наследуемости.

    s2g = h2/(1-h2) * s2e и s2_beta = s2g/c  =>  lam = c * (1 - h2) / h2.
    """
    _, _, c = center_markers(X)
    return c * (1.0 - h2) / h2


def lam_from_reml(y, X):
    """lambda из REML-оценки дисперсий (через эквивалентный GBLUP).

    Отдаёт (lam, vc), где vc — dict(sigma2_g, sigma2_e, h2, ...).
    """
    Z, _, c = center_markers(X)
    G = (Z @ Z.T) / c
    vc = reml_variance_components(y, G)
    s2_beta = vc["sigma2_g"] / c
    lam = vc["sigma2_e"] / s2_beta
    return lam, vc
