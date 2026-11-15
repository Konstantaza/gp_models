"""Явная кодировка маркеров, train-only подготовка и спектральный REML."""

import warnings
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar


def marker_values(X, coding="auto", allow_nan=False):
    """Проверка шкалы; fraction переводится в число копий того же аллеля.

    binary: 0/1 (прежняя шкала DArT); dosage: [0, 2]; fraction: [0, 1].
    auto оставлен для старых ноутбуков, но неоднозначный 0/1 предупреждает,
    а дробные значения без указания шкалы отвергаются.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or min(X.shape) == 0 or np.isinf(X).any():
        raise ValueError("Markers must be a nonempty 2D array without infinity.")
    if not allow_nan and np.isnan(X).any():
        raise ValueError("Missing markers: use prepare_markers() on train/test.")
    values = X[np.isfinite(X)]
    if not len(values):
        raise ValueError("All marker values are missing.")
    if coding == "auto":
        if np.isin(values, [0., 1.]).all():
            warnings.warn(
                "Ambiguous 0/1 markers: keeping legacy binary scale. "
                "Specify coding='binary', 'dosage' or 'fraction' explicitly.",
                UserWarning, stacklevel=2)
            coding = "binary"
        elif np.isin(values, [0., 1., 2.]).all() and (values == 2).any():
            coding = "dosage"
        else:
            raise ValueError("Fractional/ambiguous markers: specify coding explicitly.")
    if coding not in {"binary", "dosage", "fraction"}:
        raise ValueError("coding must be auto, binary, dosage or fraction.")
    upper = 2. if coding == "dosage" else 1.
    if (values < 0).any() or (values > upper).any():
        raise ValueError(f"Values outside [0, {upper:g}] for coding={coding}.")
    if coding == "binary" and not np.isin(values, [0., 1.]).all():
        raise ValueError("binary requires 0/1 calls.")
    return X * 2. if coding == "fraction" else X, coding


def allele_freq(X, coding="auto"):
    """Частота кодируемого аллеля; NA надо обработать явно до вызова."""
    A, mode = marker_values(X, coding)
    return A.mean(axis=0) / (1. if mode == "binary" else 2.)


def center_markers(X, coding="auto"):
    """(Z, p, c); binary сохраняет старую шкалу, fraction становится dosage."""
    A, mode = marker_values(X, coding)
    mean = A.mean(axis=0)
    p = mean / (1. if mode == "binary" else 2.)
    c = float(2. * np.sum(p * (1. - p)))
    if c <= 0:
        raise ValueError("No informative markers: kernel denominator is zero.")
    return A - mean, p, c


def vanraden_G(X, coding="auto"):
    """G = ZZ'/c. Для TASSEL fractions явно передать coding='fraction'."""
    Z, _, c = center_markers(X, coding)
    return Z @ Z.T / c


def prepare_markers(X_train, X_test, coding="dosage"):
    """Train-only: убрать all-NA/мономорфные SNP, заполнить NA, центрировать.

    Возвращает центрированные Z/T и исходные заполненные A/B (для Gaussian).
    Средние и маска всегда определяются только по train.
    """
    A, mode = marker_values(X_train, coding, allow_nan=True)
    # Test can be entirely missing: training means still define its imputation.
    B = np.asarray(X_test, dtype=float)
    if B.ndim != 2 or B.shape[1] != A.shape[1]:
        raise ValueError("Train/test must have the same marker columns.")
    if np.isfinite(B).any():
        B, _ = marker_values(B, mode, allow_nan=True)
    elif np.isinf(B).any():
        raise ValueError("Infinite test markers.")
    elif mode == "fraction":
        B = B * 2.
    observed = np.isfinite(A).sum(axis=0)
    mean = np.divide(np.nansum(A, axis=0), observed,
                     out=np.zeros(A.shape[1]), where=observed > 0)
    A = np.where(np.isnan(A), mean, A)
    keep = (observed > 0) & (np.var(A, axis=0) > 1e-14)
    if not keep.any():
        raise ValueError("No polymorphic training markers.")
    mean = mean[keep]
    A = A[:, keep]
    B = np.where(np.isnan(B[:, keep]), mean, B[:, keep])
    p = mean / (1. if mode == "binary" else 2.)
    c = float(2. * np.sum(p * (1. - p)))
    if c <= 0:
        raise ValueError("Zero marker normalization.")
    return dict(Z=A - mean, T=B - mean, A=A, B=B, p=p, c=c,
                mean=mean, keep=keep, n_markers=int(keep.sum()))


def fixed_design(n, F=None):
    """F включает intercept, если нужен; по умолчанию один столбец единиц."""
    F = np.ones((n, 1)) if F is None else np.asarray(F, float)
    if F.ndim == 1:
        F = F[:, None]
    if (F.ndim != 2 or F.shape[0] != n or F.shape[1] == 0
            or not np.isfinite(F).all() or n <= F.shape[1]
            or np.linalg.matrix_rank(F) != F.shape[1]):
        raise ValueError("F must be finite, full column rank, with n > n_fixed.")
    return F


def normalize_kernel(K, cross, F=None):
    """Одна train-шкала tr(P_F K)/(n-rank(F)) для обоих блоков ядра."""
    K, cross = np.asarray(K, float), np.asarray(cross, float)
    F = fixed_design(len(K), F)
    Q = np.linalg.qr(F, mode="reduced")[0]
    scale = float((np.trace(K) - np.trace(Q.T @ K @ Q))
                  / (len(K) - F.shape[1]))
    if not np.isfinite(scale) or scale <= 1e-14:
        raise ValueError("Kernel has no variation outside fixed effects.")
    return K / scale, cross / scale, scale


def kernel_eigh(K):
    """Собственные значения/векторы PSD-ядра; малый round-off обнуляется."""
    K = np.asarray(K, float)
    if (K.ndim != 2 or K.shape[0] != K.shape[1]
            or not np.isfinite(K).all() or not np.allclose(K, K.T)):
        raise ValueError("Kernel must be finite, square and symmetric.")
    d, U = eigh((K + K.T) / 2., check_finite=False)
    if d.min() < -1e-8 * max(1., abs(d).max()):
        raise ValueError("Kernel is not positive semidefinite.")
    return np.maximum(d, 0.), U


def reml_variance_components(y, K, bounds=(-8., 8.), F=None, eigen=None):
    """REML y=F beta+g+e. Старый intercept-only вызов сохранён.

    Одна eigendecomposition вместо Cholesky на каждом шаге оптимизации.
    bounds задаёт границы log(sigma2_e / sigma2_g).
    """
    y, K = np.asarray(y, float), np.asarray(K, float)
    if y.ndim != 1 or not np.isfinite(y).all() or K.shape != (len(y), len(y)):
        raise ValueError("Finite 1D y and matching square K are required.")
    F = fixed_design(len(y), F)
    d, U = kernel_eigh(K) if eigen is None else eigen
    if d.shape != (len(y),) or U.shape != K.shape:
        raise ValueError("Eigen decomposition dimensions do not match K.")
    yt, Ft = U.T @ y, U.T @ F
    df = len(y) - F.shape[1]
    if np.linalg.norm(y - F @ np.linalg.lstsq(F, y, rcond=None)[0]) <= (
            1e-12 * max(1., np.linalg.norm(y))):
        raise ValueError("No phenotype variation outside fixed effects.")

    def profile(log_lam, details=False):
        w = 1. / (d + np.exp(log_lam))
        C = Ft.T @ (w[:, None] * Ft)
        beta = np.linalg.solve(C, Ft.T @ (w * yt))
        residual = yt - Ft @ beta
        sse = float(residual @ (w * residual))
        sign, logdetC = np.linalg.slogdet(C)
        if sse <= 0 or sign <= 0:
            return np.inf
        value = .5 * (np.log(d + np.exp(log_lam)).sum()
                      + logdetC + df * np.log(sse))
        return (sse / df, beta) if details else value

    opt = minimize_scalar(profile, bounds=bounds, method="bounded",
                          options={"xatol": 1e-8})
    s2g, beta = profile(opt.x, details=True)
    lam = float(np.exp(opt.x))
    at_boundary = min(abs(opt.x - bounds[0]), abs(opt.x - bounds[1])) < .01
    return dict(sigma2_g=s2g, sigma2_e=lam * s2g, lam=lam,
                h2=1. / (1. + lam), mu=float(beta[0]), beta_fixed=beta,
                converged=bool(opt.success), at_boundary=at_boundary)
