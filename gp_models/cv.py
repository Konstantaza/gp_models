"""Кросс-валидация как в Crossa et al. (2013): случайные 70/30-разбиения,
n_reps повторов. Метрики — корреляция Пирсона pred/obs и RMSE на тесте.

Про протечку: центрирование маркеров и лямбда для RR-BLUP считаются только
по train-части каждого разбиения. В статьях иногда центрируют по всему
массиву сразу — эффект копеечный, но я делаю строго по train.
"""

import numpy as np

from .preprocessing import center_markers, reml_variance_components
from .rrblup import rrblup_fit, lam_from_reml
from .gblup import gblup_fit_predict


def random_splits(n, train_frac=0.7, n_reps=50, seed=1):
    """Генератор (train_idx, test_idx) для случайных разбиений."""
    rng = np.random.default_rng(seed)
    ntr = int(round(n * train_frac))
    for _ in range(n_reps):
        idx = rng.permutation(n)
        yield idx[:ntr], idx[ntr:]


def metrics(y_true, y_pred):
    """Отдаёт (корреляция Пирсона, RMSE)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    r = np.corrcoef(y_true, y_pred)[0, 1]
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return r, rmse


def topk_overlap(y_true, y_pred, frac=0.2):
    """Доля совпадения топ-frac линий по прогнозу и по факту.

    Селекционная метрика: какую часть лучших линий мы реально отобрали бы.
    Случайный уровень = frac (например 0.2 для топ-20%).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    k = max(1, int(round(len(y_true) * frac)))
    top_t = set(np.argsort(y_true)[-k:])
    top_p = set(np.argsort(y_pred)[-k:])
    return len(top_t & top_p) / k


def _center_by_train(Xtr, Xte):
    """Центрирование по средним train-части (test тянем тем же смещением)."""
    mean_tr = Xtr.mean(axis=0)
    return Xtr - mean_tr, Xte - mean_tr


def cv_rrblup(y, X, lam=None, n_reps=50, train_frac=0.7, seed=1,
              verbose=False):
    """CV для RR-BLUP. Если lam=None — REML-оценка внутри каждого фолда."""
    y = np.asarray(y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    rs, rmses, lams = [], [], []
    for k, (tr, te) in enumerate(random_splits(len(y), train_frac, n_reps, seed)):
        Ztr, Zte = _center_by_train(X[tr], X[te])
        if lam is None:
            lam_k, _ = lam_from_reml(y[tr], X[tr])
        else:
            lam_k = lam
        fit = rrblup_fit(y[tr], Ztr, lam_k)
        pred = fit["mu"] + Zte @ fit["beta"]
        r, rmse = metrics(y[te], pred)
        rs.append(r)
        rmses.append(rmse)
        lams.append(lam_k)
        if verbose and (k + 1) % 10 == 0:
            print(f"  rep {k + 1}/{n_reps}: r={np.mean(rs):.3f}")
    return dict(r=np.array(rs), rmse=np.array(rmses), lam=np.array(lams))


def cv_gblup(y, K, n_reps=50, train_frac=0.7, seed=1, verbose=False):
    """CV для GBLUP с готовой матрицей родства K (REML внутри каждого фолда)."""
    y = np.asarray(y, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    rs, rmses = [], []
    for k, (tr, te) in enumerate(random_splits(len(y), train_frac, n_reps, seed)):
        vc = reml_variance_components(y[tr], K[np.ix_(tr, tr)])
        pred, _ = gblup_fit_predict(y[tr], K[np.ix_(tr, tr)], K[np.ix_(te, tr)],
                                    vc["sigma2_g"], vc["sigma2_e"])
        r, rmse = metrics(y[te], pred)
        rs.append(r)
        rmses.append(rmse)
        if verbose and (k + 1) % 10 == 0:
            print(f"  rep {k + 1}/{n_reps}: r={np.mean(rs):.3f}")
    return dict(r=np.array(rs), rmse=np.array(rmses))


def cv_gibbs(y, X, fit_fn, n_reps=50, train_frac=0.7, seed=1,
             fit_kwargs=None, verbose=False):
    """CV для гиббс-моделей (BRR / BayesA).

    fit_fn(y_train, X_train, seed=..., **fit_kwargs) -> dict с 'beta' и 'mu'.
    fit_fn центрирует маркеры по train; для прогноза повторяю то же
    смещение на test.
    """
    y = np.asarray(y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    fit_kwargs = fit_kwargs or {}
    rs, rmses = [], []
    for k, (tr, te) in enumerate(random_splits(len(y), train_frac, n_reps, seed)):
        fit = fit_fn(y[tr], X[tr], seed=seed + k, **fit_kwargs)
        mean_tr = X[tr].mean(axis=0)
        pred = fit["mu"] + (X[te] - mean_tr) @ fit["beta"]
        r, rmse = metrics(y[te], pred)
        rs.append(r)
        rmses.append(rmse)
        if verbose:
            print(f"  rep {k + 1}/{n_reps}: r={r:.3f} (run.mean {np.mean(rs):.3f})")
    return dict(r=np.array(rs), rmse=np.array(rmses))
