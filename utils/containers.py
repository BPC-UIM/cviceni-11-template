"""Prepravky na vysledky fazi pipeline (PREDVYPLNENO).

Jde o orchestraci pipeline ``cviceni_11.py``, kterou student neupravuje.
Prepravky jen drzi data, ktera si faze predavaji (data, naucena sit,
predikce). Lezi mimo ``src/`` a ``dataio/``; smer zavislosti je
``utils`` -> ``src``, ``dataio`` (nikdy naopak).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src import Trainer, WindowedDataset


@dataclass
class SinusData:
    """Data sinusovky: cela rada a nahodne rozdeleni na trenovaci a testovaci cast."""

    x: np.ndarray
    y: np.ndarray
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


@dataclass
class DiabetesData:
    """Data diabetes: standardizovane priznaky a cil plus statistiky cile z trenovani."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    y_mean: float
    y_std: float


@dataclass
class DiabetesModel:
    """Vysledek faze A3: naucena sit a jeji predikce na testu v puvodnich jednotkach."""

    trainer: Trainer
    y_pred_test: np.ndarray        # (n_test,), puvodni jednotky (progrese onemocneni)


@dataclass
class SeriesData:
    """Casova rada teplot: puvodni a standardizovane casti a statistiky z trenovani."""

    train: np.ndarray              # puvodni jednotky (stupne C)
    test: np.ndarray
    train_std: np.ndarray          # standardizovane prumerem a odchylkou z trenovani
    test_std: np.ndarray
    mean: float
    std: float


@dataclass
class ForecastModel:
    """Vysledek faze B2: naucena sit a datasety testovacich oken."""

    trainer: Trainer
    test_ds: WindowedDataset
