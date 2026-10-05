"""Verejne API balicku ``src`` pro cviceni 11 (regrese a predpoved v PyTorch).

Obsahuje:

* ``MLP`` -- vicevrstva sit ``nn.Module`` s linearnim vystupem, sdilena pro
  regresi i predpoved (UKOL).
* ``Trainer`` -- trenovaci wrapper: optimizer SGD, ztrata MSE, smycka pres
  epochy a ``DataLoader``, logovani (predvyplneno); krok uceni ``_train_step``
  a perzistence ``save`` / ``load`` (UKOL).
* ``TabularDataset`` -- ``Dataset`` nad tabulkovymi daty (VZOR, predvyplneno).
* ``WindowedDataset`` -- ``Dataset`` s oknem nad signalem casove rady (UKOL).
* ``mse`` / ``mae`` / ``r2`` -- regresni metriky od nuly v NumPy (UKOL).
* ``persistence_baseline`` -- naivni predpoved "dalsi = posledni" (UKOL).
* ``ResidualAnalyzer`` -- rezidualni diagnostika regrese (predvyplneno).

**Tento soubor neupravujte.**
"""

from __future__ import annotations

from src.datasets import TabularDataset, WindowedDataset
from src.diagnostics import ResidualAnalyzer
from src.metrics import mae, mse, persistence_baseline, r2
from src.model import MLP
from src.trainer import Trainer

__all__ = [
    "MLP",
    "Trainer",
    "TabularDataset",
    "WindowedDataset",
    "mse",
    "mae",
    "r2",
    "persistence_baseline",
    "ResidualAnalyzer",
]
