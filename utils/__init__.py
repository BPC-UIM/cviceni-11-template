"""Verejne API balicku ``utils`` pro cviceni 11 (orchestrace pipeline).

Pomocny kod, ktery ridi beh ``cviceni_11.py``: vypisy hlasek, prepravky
na vysledky jednotlivych fazi a spolecna logika uceni (slozeni ``DataLoader``u,
``Trainer.fit`` a kontrola prubehu uceni). Cely balicek je **predvyplneny**,
zadny studentsky ukol, zadny ``NotImplementedError``.

Verejne API
-----------
- ``banner`` / ``faze_neni_hotova`` / ``chyba_implementace`` / ``chyba_dat`` /
  ``preskoceno`` -- jednotne hlasky pipeline
- ``SinusData`` / ``DiabetesData`` / ``DiabetesModel`` / ``SeriesData`` /
  ``ForecastModel`` -- prepravky s vysledky fazi
- ``UceniDivergovalo`` -- vyjimka pri divergenci uceni
- ``vstupni_matice`` / ``predikuj`` -- predikce site na cele sade datasetu
- ``kontrola_uceni`` -- divergence, neklesajici ztrata, preuceni (prahy z ``config.yaml``)
- ``nauc_model`` -- ``DataLoader``y + ``Trainer.fit`` + ``kontrola_uceni``

**Tento soubor neupravujte.**
"""

from __future__ import annotations

from utils.containers import (
    DiabetesData,
    DiabetesModel,
    ForecastModel,
    SeriesData,
    SinusData,
)
from utils.reporting import (
    banner,
    chyba_dat,
    chyba_implementace,
    faze_neni_hotova,
    preskoceno,
)
from utils.training import (
    UceniDivergovalo,
    kontrola_uceni,
    nauc_model,
    predikuj,
    vstupni_matice,
)

__all__ = [
    "banner",
    "faze_neni_hotova",
    "chyba_implementace",
    "chyba_dat",
    "preskoceno",
    "SinusData",
    "DiabetesData",
    "DiabetesModel",
    "SeriesData",
    "ForecastModel",
    "UceniDivergovalo",
    "vstupni_matice",
    "predikuj",
    "kontrola_uceni",
    "nauc_model",
]
