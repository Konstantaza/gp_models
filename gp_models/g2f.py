"""Небольшой загрузчик числового TASSEL G2F; VCF не требуется."""

from pathlib import Path
import numpy as np
import pandas as pd


def load_g2f_numeric(path):
    """Читает <Numeric>/<Marker>, возвращает DataFrame Hybrid×SNP.

    В файле допускаются только 0, 0.5, 1, NA. Возвращаются 0/1/2 и NaN:
    число копий кодируемого аллеля, не обязательно ALT. Нули не пропуски.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing G2F numerical genotype file: {path}")
    with path.open(encoding="utf-8-sig") as handle:
        if handle.readline().strip() != "<Numeric>":
            raise ValueError("Expected <Numeric> on line 1.")
        header = handle.readline().split()
        if not header or header[0] != "<Marker>":
            raise ValueError("Expected <Marker> on line 2.")
        markers = header[1:]
        if not markers or len(markers) != len(set(markers)):
            raise ValueError("Empty or duplicated marker names.")
        names, rows = [], []
        allowed = {"0": 0., "0.5": 1., "1": 2., "NA": np.nan}
        for line_no, line in enumerate(handle, 3):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != len(markers) + 1:
                raise ValueError(f"Line {line_no}: wrong number of fields.")
            names.append(fields[0])
            try:
                rows.append(np.array([allowed[x] for x in fields[1:]], dtype=float))
            except KeyError as exc:
                raise ValueError(f"Line {line_no}: unexpected token {exc.args[0]!r}.") from exc
    if not rows or len(names) != len(set(names)):
        raise ValueError("No genotype rows or duplicated Hybrid names.")
    return pd.DataFrame(np.stack(rows), index=pd.Index(names, name="Hybrid"),
                        columns=markers)


def regression_metrics(y_true, y_pred):
    """Без молчаливого удаления пар. r/R² неопределённы → NaN."""
    y, p = np.asarray(y_true, float), np.asarray(y_pred, float)
    if y.ndim != 1 or p.shape != y.shape or not len(y):
        raise ValueError("Nonempty matching 1D true/pred arrays required.")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Metrics require finite true/pred values.")
    err = y - p
    sst = float(np.sum((y - y.mean())**2))
    r = (float(np.corrcoef(y, p)[0, 1])
         if len(y) > 1 and np.ptp(y) > 0 and np.ptp(p) > 0 else np.nan)
    return dict(r=r, rmse=float(np.sqrt(np.mean(err**2))),
                mae=float(np.mean(abs(err))),
                r2=float(1. - np.sum(err**2)/sst) if np.ptp(y) > 0 else np.nan)
