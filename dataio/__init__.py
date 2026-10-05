"""Verejne API balicku ``dataio`` pro cviceni 11.

Nacteni a generovani dat pro obe casti cviceni, rozdeleni dat (nahodne pro
regresi, chronologicke pro casovou radu), typovana konfigurace a vykreslovani.

Verejne API
-----------
- ``load_diabetes_data`` -- tabulkova regrese (diabetes, 10 priznaku), standardizovano
- ``make_sinusoid`` -- zasumena sinusovka pro 1D regresi
- ``load_temperature_series`` -- denni minimalni teploty (Melbourne) jako 1D pole
- ``random_split`` -- nahodne rozdeleni nezavislych vzorku (cast A)
- ``chronological_split`` -- rozdeleni casove rady BEZ michani (cast B)
- ``TEMPERATURE_CSV`` -- vychozi cesta k CSV s teplotami
- ``load_config`` / ``validate_config`` -- typovana konfigurace nad ``config.yaml``
- ``PathsConfig`` / ``SinusoidConfig`` / ``TrainingConfig`` / ``RegressionConfig`` /
  ``ForecastConfig`` / ``DiagnosticsConfig`` / ``ExperimentConfig`` -- dataclassy
  konfigurace (vse nastavitelne je v ``config.yaml``)
- ``plot_regression_fit`` -- prolozeni 1D dat krivkou site
- ``plot_predicted_vs_actual`` -- predikce proti skutecnosti (vice priznaku)
- ``plot_loss_curve`` -- krivka uceni (trenovaci a validacni ztrata)
- ``plot_forecast`` -- predpoved a skutecnost v case
- ``plot_residuals_in_time`` -- rezidua predpovedi v case

Cely balicek ``dataio/`` je v tomto cviceni **predvyplneny** -- zadny
studentsky ukol, zadny ``NotImplementedError``.

**Tento soubor neupravujte.**
"""

from __future__ import annotations

from dataio.config_manager import (
    DiagnosticsConfig,
    ExperimentConfig,
    ForecastConfig,
    PathsConfig,
    RegressionConfig,
    SinusoidConfig,
    TrainingConfig,
    load_config,
    validate_config,
)
from dataio.loaders import (
    TEMPERATURE_CSV,
    chronological_split,
    load_diabetes_data,
    load_temperature_series,
    make_sinusoid,
    random_split,
)
from dataio.plotting import (
    plot_forecast,
    plot_loss_curve,
    plot_predicted_vs_actual,
    plot_regression_fit,
    plot_residuals_in_time,
)

__all__ = [
    "load_diabetes_data",
    "make_sinusoid",
    "load_temperature_series",
    "random_split",
    "chronological_split",
    "TEMPERATURE_CSV",
    "load_config",
    "validate_config",
    "PathsConfig",
    "SinusoidConfig",
    "TrainingConfig",
    "RegressionConfig",
    "ForecastConfig",
    "DiagnosticsConfig",
    "ExperimentConfig",
    "plot_regression_fit",
    "plot_predicted_vs_actual",
    "plot_loss_curve",
    "plot_forecast",
    "plot_residuals_in_time",
]
