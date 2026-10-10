"""BayesA — то же, что BRR, но у каждого маркера СВОЯ дисперсия.

    beta_j | s2b_j ~ N(0, s2b_j),   s2b_j ~ Inv-chi2(dfb, Sb)

Если проинтегрировать s2b_j, на beta_j получается t-приор с тяжёлыми
хвостами: мелкие эффекты прижимаются к нулю ещё сильнее, чем в BRR,
а крупным (QTL) даётся свобода. Общей дисперсии на все маркеры тут нет.

Полные условные почти как в BRR:
    beta_j | . ~ N(rhs/lhs_j, s2e/lhs_j),  lhs_j = z_j'z_j + s2e/s2b_j
    s2b_j  | . ~ Inv-chi2(dfb + 1, Sb + beta_j^2)   (обновляю после каждого j)
mu и s2e — как в BRR. По коду отличие от brr.py — четыре строчки в цикле
по маркерам.
"""

import numpy as np
from numba import njit

from .preprocessing import center_markers
from .brr import prior_scales


@njit(cache=True)
def _gibbs_bayesa_kernel(Zt, y, n_iter, burn_in, thin,
                         dfb, Sb, dfe, Se, s2b0, s2e0, seed):
    """Ядро Гиббса для BayesA. Zt: (p, n) центрированные маркеры."""
    np.random.seed(seed)
    p, n = Zt.shape
    x2 = np.empty(p)
    for j in range(p):
        x2[j] = np.dot(Zt[j], Zt[j])

    b = np.zeros(p)
    s2b = np.full(p, s2b0)          # теперь вектор: своя дисперсия на маркер
    mu = np.mean(y)
    e = y - mu
    s2e = s2e0

    n_keep = (n_iter - burn_in) // thin
    b_sum = np.zeros(p)
    b_sq = np.zeros(p)
    s2b_sum = np.zeros(p)
    trace = np.empty((n_keep, 2))   # mu, s2e
    ik = 0
    for it in range(n_iter):
        # mu — как в BRR
        for i in range(n):
            e[i] += mu
        mu = np.mean(e) + np.sqrt(s2e / n) * np.random.standard_normal()
        for i in range(n):
            e[i] -= mu
        # маркеры по одному, со своей дисперсией у каждого
        for j in range(p):
            xj = Zt[j]
            bj = b[j]
            for i in range(n):
                e[i] += xj[i] * bj            # вернуть вклад маркера в остатки
            rhs = np.dot(xj, e)
            lam_j = s2e / s2b[j]              # своя лямбда на маркер
            lhs = x2[j] + lam_j
            bnew = rhs / lhs + np.sqrt(s2e / lhs) * np.random.standard_normal()
            for i in range(n):
                e[i] -= xj[i] * bnew          # вычесть вклад уже с новым beta
            b[j] = bnew
            # и сразу обновить его индивидуальную дисперсию
            s2b[j] = (Sb + bnew * bnew) / \
                (2.0 * np.random.gamma((dfb + 1.0) / 2.0, 1.0))
        # s2e — как в BRR
        s2e = (Se + np.sum(e * e)) / (2.0 * np.random.gamma((dfe + n) / 2.0, 1.0))
        if it >= burn_in and (it - burn_in) % thin == 0:
            b_sum += b
            b_sq += b * b
            s2b_sum += s2b
            trace[ik, 0] = mu
            trace[ik, 1] = s2e
            ik += 1
    return b_sum, b_sq, s2b_sum, trace, ik


def bayesa_fit(y, X, n_iter=12000, burn_in=2000, thin=5,
               dfb=4.0, dfe=4.0, R2=0.5, seed=1):
    """Обёртка над ядром (как brr_fit). Дополнительно отдаёт sigma2_bj —
    posterior-средние индивидуальных дисперсий маркеров."""
    y = np.asarray(y, dtype=np.float64)
    Z, _, c = center_markers(X)
    Zt = np.ascontiguousarray(Z.T)
    Sb, Se, _ = prior_scales(y, X, R2=R2, dfb=dfb, dfe=dfe)
    s2b0 = R2 * y.var() / c
    s2e0 = (1.0 - R2) * y.var()
    b_sum, b_sq, s2b_sum, trace, ik = _gibbs_bayesa_kernel(
        Zt, y, n_iter, burn_in, thin, dfb, Sb, dfe, Se, s2b0, s2e0, seed)
    trace = trace[:ik]
    beta = b_sum / ik
    beta_sd = np.sqrt(np.maximum(b_sq / ik - beta**2, 0.0))
    return dict(beta=beta, beta_sd=beta_sd, sigma2_bj=s2b_sum / ik,
                mu=float(trace[:, 0].mean()),
                sigma2_e=float(trace[:, 1].mean()),
                trace=trace, gebv=float(trace[:, 0].mean()) + Z @ beta,
                c=c, Sb=Sb, Se=Se)
