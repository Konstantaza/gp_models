"""BRR (Bayesian Ridge Regression) — байесовский вариант RR-BLUP.

Модель та же, но на дисперсии ставятся приоры, и они оцениваются вместе
с эффектами гиббс-сэмплером:
    beta_j | s2b ~ N(0, s2b)     (одна общая дисперсия на все маркеры)
    s2b ~ Inv-chi2(dfb, Sb),  s2e ~ Inv-chi2(dfe, Se),  mu — плоский приор

Полные условные, по которым ходит Гиббс:
    beta_j | . ~ N(rhs/lhs, s2e/lhs),  rhs = z_j'(e + z_j beta_j),
                                       lhs = z_j'z_j + s2e/s2b
    s2b | . ~ Inv-chi2(dfb + p, Sb + sum_j beta_j^2)
    s2e | . ~ Inv-chi2(dfe + n, Se + sum_i e_i^2)
    mu  | . ~ N(mean(e + mu), s2e/n)

Inv-chi2(df, S) сэмплирую как S / chi2(df), а chi2(df) = 2*Gamma(df/2, 1).

Приорные шкалы — правило из пакета BGLR: у Inv-chi2(df, S) мода равна
S/(df+2), поэтому S подбираю так, чтобы мода s2b попала в R2*var(y)/c
(доля R2 дисперсии фенотипа, размазанная по маркерам, c = 2*sum p(1-p)),
а мода s2e — в (1-R2)*var(y).
"""

import numpy as np
from numba import njit

from .preprocessing import center_markers


def prior_scales(y, X, R2=0.5, dfb=4.0, dfe=4.0):
    """Sb, Se по правилу BGLR. Отдаёт (Sb, Se, c)."""
    y = np.asarray(y, dtype=np.float64)
    _, _, c = center_markers(X)
    vy = y.var(ddof=1)
    Sb = (dfb + 2.0) * R2 * vy / c
    Se = (dfe + 2.0) * (1.0 - R2) * vy
    return Sb, Se, c


@njit(cache=True)
def _gibbs_brr_kernel(Zt, y, n_iter, burn_in, thin,
                      dfb, Sb, dfe, Se, s2b0, s2e0, seed):
    """Ядро Гиббса. Zt — центрированные маркеры в виде (p, n): строка Zt[j]
    лежит в памяти подряд, так цикл по маркерам намного быстрее."""
    np.random.seed(seed)
    p, n = Zt.shape
    x2 = np.empty(p)
    for j in range(p):
        x2[j] = np.dot(Zt[j], Zt[j])   # z_j'z_j не меняются — считаю один раз

    b = np.zeros(p)
    mu = np.mean(y)
    e = y - mu                         # текущие остатки y - mu - Zb
    s2b, s2e = s2b0, s2e0

    n_keep = (n_iter - burn_in) // thin
    b_sum = np.zeros(p)
    b_sq = np.zeros(p)
    trace = np.empty((n_keep, 3))      # mu, s2b, s2e
    ik = 0
    for it in range(n_iter):
        # mu
        for i in range(n):
            e[i] += mu
        mu = np.mean(e) + np.sqrt(s2e / n) * np.random.standard_normal()
        for i in range(n):
            e[i] -= mu
        # маркеры по одному: вернуть вклад -> обновить -> вычесть с новым b
        lam = s2e / s2b
        for j in range(p):
            xj = Zt[j]
            bj = b[j]
            for i in range(n):
                e[i] += xj[i] * bj
            rhs = np.dot(xj, e)
            lhs = x2[j] + lam
            bnew = rhs / lhs + np.sqrt(s2e / lhs) * np.random.standard_normal()
            for i in range(n):
                e[i] -= xj[i] * bnew
            b[j] = bnew
        # дисперсии
        s2b = (Sb + np.sum(b * b)) / (2.0 * np.random.gamma((dfb + p) / 2.0, 1.0))
        s2e = (Se + np.sum(e * e)) / (2.0 * np.random.gamma((dfe + n) / 2.0, 1.0))
        # накопление (после burn-in, с прореживанием)
        if it >= burn_in and (it - burn_in) % thin == 0:
            b_sum += b
            b_sq += b * b
            trace[ik, 0] = mu
            trace[ik, 1] = s2b
            trace[ik, 2] = s2e
            ik += 1
    return b_sum, b_sq, trace, ik


def brr_fit(y, X, n_iter=12000, burn_in=2000, thin=5,
            dfb=4.0, dfe=4.0, R2=0.5, seed=1):
    """Обёртка над ядром: центрирует маркеры, считает приоры, гоняет Гиббса.

    Отдаёт dict(beta, beta_sd, mu, sigma2_b, sigma2_e, trace, gebv).
    """
    y = np.asarray(y, dtype=np.float64)
    Z, _, c = center_markers(X)
    Zt = np.ascontiguousarray(Z.T)
    Sb, Se, _ = prior_scales(y, X, R2=R2, dfb=dfb, dfe=dfe)
    s2b0 = R2 * y.var() / c
    s2e0 = (1.0 - R2) * y.var()
    b_sum, b_sq, trace, ik = _gibbs_brr_kernel(
        Zt, y, n_iter, burn_in, thin, dfb, Sb, dfe, Se, s2b0, s2e0, seed)
    trace = trace[:ik]
    beta = b_sum / ik
    beta_sd = np.sqrt(np.maximum(b_sq / ik - beta**2, 0.0))
    return dict(beta=beta, beta_sd=beta_sd,
                mu=float(trace[:, 0].mean()),
                sigma2_b=float(trace[:, 1].mean()),
                sigma2_e=float(trace[:, 2].mean()),
                trace=trace, gebv=float(trace[:, 0].mean()) + Z @ beta,
                c=c, Sb=Sb, Se=Se)
