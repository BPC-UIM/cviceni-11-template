"""Spolecna logika uceni pro obe casti pipeline (PREDVYPLNENO).

Jde o orchestraci pipeline ``cviceni_11.py``, kterou student neupravuje.
Lezi mimo ``src/`` (studentske ukoly) a ``dataio/`` (data, konfigurace,
grafy); smer zavislosti je ``utils`` -> ``src``, ``dataio`` (nikdy naopak).

Vsechny prahy kontrol (divergence, preuceni) se ctou z sekce ``diagnostics``
v ``config.yaml`` (``DiagnosticsConfig``), nejsou to konstanty modulu.
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dataio import DiagnosticsConfig, TrainingConfig
from src import Trainer
from src.trainer import LOG_FILE_NAME


class UceniDivergovalo(RuntimeError):
    """Uceni skoncilo nekonecnou ztratou nebo ztrata vzrostla radove oproti 1. epose."""


def vstupni_matice(dataset: Dataset) -> torch.Tensor:
    """Vrati vsechny vstupy datasetu jako jednu matici ``(n, n_features)``.

    Parametry
    ---------
    dataset : Dataset
        Dataset vracejici dvojice ``(x, y)``.

    Navratova hodnota
    -----------------
    torch.Tensor
        Vstupy vsech vzorku slozene do jedne davky.
    """
    x_all, _ = next(iter(DataLoader(dataset, batch_size=len(dataset), shuffle=False)))
    return x_all


def predikuj(trainer: Trainer, dataset: Dataset) -> np.ndarray:
    """Predikce site na cele sade datasetu jako 1D pole NumPy ``(n,)``.

    Parametry
    ---------
    trainer : Trainer
        Trener s naucenym modelem.
    dataset : Dataset
        Dataset, jehoz vstupy se predikuji.

    Navratova hodnota
    -----------------
    np.ndarray
        Predikce modelu (v jednotkach, ve kterych se sit ucila).
    """
    return trainer.predict(vstupni_matice(dataset)).detach().numpy().reshape(-1)


def kontrola_uceni(
    history: dict[str, list[float]],
    diagnostics: DiagnosticsConfig,
    section: str | None = None,
    ) -> None:
    """Zkontroluje prubeh uceni: divergence, neklesajici ztrata, preuceni.

    * Nekonecna ztrata nebo konecna trenovaci ztrata vetsi nez
      ``diagnostics.divergence_factor`` nasobek ztraty v 1. epose
      -> ``UceniDivergovalo``.
    * Vzdy se vypise nejnizsi validacni ztrata a jeji epocha.
    * Neklesla-li trenovaci ztrata, upozorni se na chybu v ``_train_step``.
    * Preuceni: konecna validacni ztrata je vyssi nez
      ``(1 + diagnostics.overfit_rise)`` nasobek minima A minimum nastalo pred
      ``diagnostics.overfit_min_fraction`` epoch (a trenovaci ztrata pritom klesla).

    Parametry
    ---------
    history : dict[str, list[float]]
        Historie z ``Trainer.fit`` (klice ``epoch``, ``train_loss``, ``val_loss``).
    diagnostics : DiagnosticsConfig
        Prahy kontrol z ``config.yaml`` (sekce ``diagnostics``).
    section : str | None, optional
        Sekce ``config.yaml``, ze ktere pochazi nastaveni uceni (``"regression"``
        nebo ``"forecast"``); rada pri preuceni pak jmenuje presny klic
        (napr. ``regression.epochs``). ``None`` = obecne ``epochs``.

    Vyjimky
    -------
    UceniDivergovalo
        Pri divergenci uceni.
    """
    train, val = history["train_loss"], history["val_loss"]
    if not (np.all(np.isfinite(train)) and np.all(np.isfinite(val))) \
            or train[-1] > diagnostics.divergence_factor * train[0]:
        raise UceniDivergovalo("ztrata divergovala -- zmensete lr")

    pocet = len(val)
    index_min = int(np.argmin(val))
    epocha_min = index_min + 1
    print(f"  nejnizsi validacni ztrata {val[index_min]:.4f} v epose {epocha_min} z {pocet}")
    if train[-1] >= train[0]:
        print("  [POZOR] Trenovaci ztrata neklesa -- zkontrolujte _train_step "
              "(zero_grad, poradi kroku) nebo zmensete lr.")
    elif (val[-1] > (1.0 + diagnostics.overfit_rise) * val[index_min]
          and epocha_min < diagnostics.overfit_min_fraction * pocet):
        print(f"  [POZOR] Validacni ztrata od epochy {epocha_min} roste, "
              "trenovaci dal klesa -- sit se preucuje.")
        klic = f"{section}.epochs" if section else "epochs"
        print(f"  -> Zkuste v config.yaml snizit {klic} priblizne na {epocha_min} "
              "a porovnejte metriky.")
        print("     (Citlivost varovani nastavuji diagnostics.overfit_rise "
              "a diagnostics.overfit_min_fraction.)")


def nauc_model(
    model: torch.nn.Module,
    train_ds: Dataset,
    val_ds: Dataset,
    training: TrainingConfig,
    logs_dir: str,
    diagnostics: DiagnosticsConfig,
    section: str | None = None,
    ) -> tuple[Trainer, dict[str, list[float]]]:
    """Slozi ``DataLoader``y, vytvori ``Trainer``, spusti ``fit`` a zkontroluje uceni.

    Trenovaci loader micha poradi VZORKU (u casove rady poradi oken, rada je
    ale rozdelena chronologicky uz drive), validacni loader nemicha.

    Parametry
    ---------
    model : nn.Module
        Neucena sit.
    train_ds, val_ds : Dataset
        Trenovaci a validacni (testovaci) data.
    training : TrainingConfig
        Hyperparametry uceni (``lr``, ``epochs``, ``batch_size``, ``seed``).
    logs_dir : str
        Adresar pro soubor logu (``paths.logs_dir``).
    diagnostics : DiagnosticsConfig
        Prahy kontrol z ``config.yaml``.
    section : str | None, optional
        Sekce ``config.yaml`` s nastavenim uceni (pro presne hlasky), viz
        ``kontrola_uceni``.

    Navratova hodnota
    -----------------
    tuple[Trainer, dict[str, list[float]]]
        Trener po uceni a historie ztrat.

    Vyjimky
    -------
    UceniDivergovalo
        Pri divergenci uceni.
    """
    train_loader = DataLoader(train_ds, batch_size=training.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=training.batch_size, shuffle=False)
    trainer = Trainer(model, training, log_dir=logs_dir)
    start = time.perf_counter()
    history = trainer.fit(train_loader, val_loader)
    print(f"  doba uceni: {time.perf_counter() - start:.1f} s; "
          f"uplny zaznam kazde epochy: {os.path.join(logs_dir, LOG_FILE_NAME)}")
    kontrola_uceni(history, diagnostics, section)
    return trainer, history
