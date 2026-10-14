"""Загрузка датасетов.

Пшеница CIMMYT из Crossa et al. (2010, Genetics 186:713-724) — файл
wheat.RData из R-пакета BGLR. Внутри:
  X     599 линий x 1279 маркеров DArT (0/1); авторы уже отфильтровали
        по MAF >= 0.05 и заимпутировали пропуски;
  Y     урожайность в 4 мега-средах, стандартизирована внутри среды;
  A     матрица родства по родословной;
  sets  разбиение на 10 фолдов из статьи.

Кукуруза CIMMYT — Cuevas et al. (2017, G3 7:1833-1847), файл с
data.cimmyt.org (hdl:11529/10907). Сырых маркеров там нет, зато есть
посчитанная авторами матрица родства G и таблица фенотипов
(Gid, Env, Rep, Yield, ASI, PH).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_wheat(data_dir=None):
    """Читает wheat.RData, отдаёт dict с X, Y, A, sets и именами линий."""
    d = Path(data_dir) if data_dir else DATA_DIR
    r = pyreadr.read_r(str(d / "wheat.RData"))
    Xdf, Ydf, Adf = r["wheat.X"], r["wheat.Y"], r["wheat.A"]
    return dict(
        X=Xdf.to_numpy(dtype=np.float64),
        Y=Ydf.to_numpy(dtype=np.float64),
        A=Adf.to_numpy(dtype=np.float64),
        sets=r["wheat.sets"].to_numpy().ravel().astype(int),
        line_ids=Xdf.index.to_numpy(),
        env_names=list(Ydf.columns),
    )


def load_maize(data_dir=None):
    """Читает DATASET1to3_Maize.rdata, отдаёт dict(pheno, G, g_ids)."""
    d = Path(data_dir) if data_dir else DATA_DIR
    r = pyreadr.read_r(str(d / "DATASET1to3_Maize.rdata"))
    pheno = r["Maize.Data"].copy()
    Gdf = r["G"]
    return dict(pheno=pheno, G=Gdf.to_numpy(dtype=np.float64),
                g_ids=Gdf.index.to_numpy())


def load_dryad_maize(data_dir=None):
    """Кукуруза CIMMYT с Dryad (мульти-поколенческий набор 2017-2020).

    Отдаёт dict(pheno, X, marker_names): pheno — таблица
    [GID, Year, GY/PH/AD x Optimal/Drought], X — маркеры 0/1 (4612 шт.),
    строки X выровнены по порядку строк pheno (по GID).
    """
    d = Path(data_dir) if data_dir else DATA_DIR
    ph = pd.read_csv(d / "Pheno_data.csv")
    g = pd.read_csv(d / "Geno_data.csv").set_index("GID")
    g = g.loc[ph["GID"]]   # выравниваю маркеры под порядок фенотипов
    return dict(pheno=ph, X=g.to_numpy(dtype=np.float64),
                marker_names=np.array(g.columns))
