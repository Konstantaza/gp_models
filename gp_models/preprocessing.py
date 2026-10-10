"""Подготовка маркеров и REML-оценка компонент дисперсии.

Центрирование:  Z_ij = X_ij - 2 p_j,  где p_j — частота аллеля.
Матрица родства ВанРадена:  G = Z Z' / c,  c = 2 * sum_j p_j (1 - p_j).
Делитель c нужен, чтобы дисперсии в двух параметризациях сходились:
sigma2_g = c * sigma2_beta.

REML для модели y = 1 mu + g + e, g ~ N(0, K s2g), e ~ N(0, I s2e).
n у нас маленькое (сотни), поэтому просто минимизирую остаточное
правдоподобие по lambda = s2e/s2g с Холецким внутри — без спектральных
трюков, их имеет смысл добавлять на больших n.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize_scalar


def allele_freq(X):
    """Частоты аллеля '1' по столбцам (работает и для 0/1, и для 0/1/2)."""
    return np.asarray(X, dtype=np.float64).mean(axis=0) / (
        2.0 if np.nanmax(X) > 1.5 else 1.0
    )


def center_markers(X):
    """Отдаёт (Z, p, c): центрированную матрицу, частоты и нормировку c."""
    X = np.asarray(X, dtype=np.float64)
    p = X.mean(axis=0) / (2.0 if X.max() > 1.5 else 1.0)
    Z = X - X.mean(axis=0)          # то же, что X - 2p для кода 0/1/2
    c = 2.0 * np.sum(p * (1.0 - p))
    return Z, p, c


def vanraden_G(X):
    """Матрица родства ВанРадена из сырых маркеров (0/1 или 0/1/2)."""
    Z, p, c = center_markers(X)
    return (Z @ Z.T) / c


def reml_variance_components(y, K, bounds=(-8.0, 8.0)):
    """REML-оценки (s2g, s2e) для y = 1 mu + g + e, g ~ N(0, K s2g).

    Пишу V = K + lam*I (то есть V с точностью до множителя s2g) и
    минимизирую профильное остаточное лог-правдоподобие по lam:
        ll = -0.5 * [ logdet(V) + log(1' V^-1 1) + (n-1) * log(y' P y) ]
        P = V^-1 - V^-1 1 (1' V^-1 1)^-1 1' V^-1
    s2g на оптимуме = y'Py/(n-1). Константу опускаю — на минимум она
    не влияет.
    """
    y = np.asarray(y, dtype=np.float64)
    n = y.shape[0]
    one = np.ones(n)
    I = np.eye(n)

    def neg_ll(log_lam):
        V = K + np.exp(log_lam) * I
        c_fac = cho_factor(V)
        Vi_1y = cho_solve(c_fac, np.column_stack([one, y]))
        Vi1, Viy = Vi_1y[:, 0], Vi_1y[:, 1]
        logdetV = 2.0 * np.sum(np.log(np.diag(c_fac[0])))
        xVix = one @ Vi1
        yPy = y @ Viy - (one @ Viy) ** 2 / xVix
        return 0.5 * (logdetV + np.log(xVix) + (n - 1) * np.log(yPy))

    opt = minimize_scalar(neg_ll, bounds=bounds, method="bounded",
                          options={"xatol": 1e-8})
    lam = np.exp(opt.x)
    V = K + lam * I
    c_fac = cho_factor(V)
    Vi_1y = cho_solve(c_fac, np.column_stack([one, y]))
    Vi1, Viy = Vi_1y[:, 0], Vi_1y[:, 1]
    xVix = one @ Vi1
    yPy = y @ Viy - (one @ Viy) ** 2 / xVix
    s2g = yPy / (n - 1)
    s2e = lam * s2g
    mu = (one @ Viy) / xVix
    return dict(sigma2_g=s2g, sigma2_e=s2e, lam=lam, h2=s2g / (s2g + s2e),
                mu=mu, converged=opt.success)
