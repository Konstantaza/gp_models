"""Random forest adapter to scikit-learn, not a from-scratch tree algorithm."""
import numpy as np
from sklearn.ensemble import RandomForestRegressor


def random_forest_fit(y, X, seed=1, n_estimators=200, min_samples_leaf=5,
                      max_features=.3, n_jobs=2):
    X, y = np.asarray(X, float), np.asarray(y, float)
    keep = X.var(axis=0) > 0
    if not keep.any():
        raise ValueError("No polymorphic training markers.")
    model = RandomForestRegressor(n_estimators=n_estimators,
                                  min_samples_leaf=min_samples_leaf,
                                  max_features=max_features, n_jobs=n_jobs,
                                  random_state=seed)
    model.fit(X[:, keep], y)
    return dict(model=model, keep=keep, mu=float(y.mean()), n_markers=int(keep.sum()))


def random_forest_predict(fit, X):
    return fit["model"].predict(np.asarray(X, float)[:, fit["keep"]])
