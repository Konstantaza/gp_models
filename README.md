# Геномное предсказание: RR-BLUP, GBLUP, BRR, BayesA

Реализую основные модели геномного прогнозирования на Python (numpy + numba).

## Данные

- **Пшеница** — Crossa et al. 2010, *Genetics* 186:713–724
  (doi:10.1534/genetics.110.118521). 599 линий × 1279 маркеров DArT,
  урожайность в 4 мега-средах, родословная матрица A. Файл `wheat.RData`
  взят из R-пакета BGLR (github.com/gdlc/BGLR-R).
- **Кукуруза** — Cuevas et al. 2017, *G3* 7:1833–1847
  (doi:10.1534/g3.117.300095), data.cimmyt.org (hdl:11529/10907).
  309 DH-линий в 3 средах + готовая матрица родства G.

Почему маркерные модели на пшенице, а не на кукурузе: сырые маркеры
CIMMYT-кукурузы лежат под запросом (restricted), а пшеничный набор — это
ровно те данные, на которых Crossa et al. (2010) сравнивали Bayesian
LASSO и RKHS. Маленький, чистый, удобный. Кукуруза идёт в дело в
ноутбуке 06 (GBLUP на готовой G).

## Запуск (VSCode)

```bash
python -m venv .venv
source .venv/bin/activate        # на Windows: .venv\Scripts\activate
pip install -r requirements.txt  # numba просит numpy < 2.3, уже закреплено
```

В VSCode поставить расширения Python и Jupyter, выбрать интерпретатор
`.venv` (Ctrl+Shift+P → "Python: Select Interpreter") и выполнять
ноутбуки по порядку.

## Структура

```
gp_models/
├── data/                    # wheat.RData, DATASET1to3_Maize.rdata
├── gp_models/
│   ├── data.py              # загрузка датасетов
│   ├── preprocessing.py     # центрирование, G ВанРадена, REML
│   ├── rrblup.py            # RR-BLUP (смешанные уравнения)
│   ├── gblup.py             # GBLUP + прогноз через родство
│   ├── brr.py               # байесовский ridge, гиббс-сэмплер (numba)
│   ├── bayesa.py            # BayesA (своя дисперсия на маркер)
│   └── cv.py                # кросс-валидация 70/30
├── notebooks/
│   ├── 01_data_exploration.ipynb   # данные, частоты, матрицы G и A
│   ├── 02_rrblup_gblup.ipynb       # RR-BLUP, эквивалентность с GBLUP
│   ├── 03_brr_gibbs.ipynb          # гиббс-сэмплер, диагностика
│   ├── 04_bayesa.ipynb             # BayesA, сравнение с BRR
│   ├── 05_cv_comparison.ipynb      # CV 70/30 × 50: три модели
│   └── 06_gblup_maize.ipynb        # GBLUP на кукурузе CIMMYT
├── figures/                 # графики из ноутбуков
├── results/                 # cv_results.csv
└── requirements.txt
```

## Как проверял, что код правильный

- GEBV из RR-BLUP и из GBLUP совпадают до ~1e-15 (это одна модель в двух
  параметризациях);
- мой RR-BLUP совпадает со sklearn Ridge при α = λ (corr = 1.0);
- posterior mean β из BRR ≈ β̂ из RR-BLUP (corr ≈ 0.998);
- REML восстанавливает дисперсии на симулированных данных;
- в кукурузе контроль с перемешанной G даёт r ≈ 0 — прогноз держится на
  родстве, а не на протечке.

## Чем пользовались авторы оригинальных статей

| Статья | Софт |
|---|---|
| Crossa et al. 2010, 2013 | R-пакет BLR (предок BGLR, github.com/gdlc/BGLR-R); GBS в 2013 — через TASSEL |
| Pérez-Rodríguez et al. 2012 | BLR v1.2, AlphaBayes, FBM (Radford Neal) для BRNN |
| Alemu et al. 2024 | R-скрипт в Supplemental File 2 статьи (doi:10.1016/j.molp.2024.02.006) |

Про сам BGLR: Pérez & de los Campos 2014, doi:10.1534/genetics.114.164442.
