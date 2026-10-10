"""gp_models — модели геномного прогнозирования, написанные с нуля.

data           загрузка пшеницы (BGLR) и кукурузы (Cuevas et al. 2017)
preprocessing  центрирование маркеров, G ВанРадена, REML-оценка дисперсий
rrblup         RR-BLUP через смешанные уравнения (замкнутая форма)
gblup          GBLUP через матрицу родства
brr            байесовский ridge, гиббс-сэмплер (numba)
bayesa         BayesA — своя дисперсия на каждый маркер
cv             кросс-валидация 70/30 x n_reps
"""

from . import data, preprocessing, rrblup, gblup, brr, bayesa, cv  # noqa: F401
