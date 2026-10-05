"""Metriky regrese a naivni baseline -- NumPy od nuly (bez sklearn).

U regrese nemame "spravne / spatne" jako u klasifikace, merime velikost chyby
`e_i = y_true_i - y_pred_i`:

- `mse` - prumer ctvercu chyb (velke chyby tresta nejvic, jednotka je jednotka^2),
- `mae` - prumer absolutnich hodnot chyb (jednotka je stejna jako u dat),
- `r2`  - koeficient determinace, **podil vysvetleneho rozptylu**:
  `R^2 = 1` je dokonala predikce, `R^2 = 0` znamena, ze model je stejne dobry
  jako predikce konstantou = prumerem skutecnych hodnot, a `R^2 < 0` znamena,
  ze je model **horsi nez prumer** (a tedy k nicemu).

`persistence_baseline` je nejjednodussi mozna predpoved casove rady: "zitra bude
stejne jako dnes". Je to obdoba "accuracy klame" z cviceni 6 -- samotne cislo
(napr. MSE) nic nerika, dokud ho neporovname s tim, co zvladne trivialni
pravidlo. Neuronova sit, ktera baseline neporazi, se nenaucila nic uziteckeho.

Pozor na tvary: NumPy pri odecitani poli `(n,)` a `(n, 1)` ticho broadcastuje
na `(n, n)` a spocita nesmysl (stejna past jako u `nn.MSELoss`). Proto vsechny
metriky nejdriv overuji, ze oba vstupy maji shodny tvar. Pipeline predava
1D pole `(n,)`.
"""

from __future__ import annotations

import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Vypocte stredni kvadratickou chybu, Mean Squared Error (UKOL).

    Definice
    --------
    MSE = (1 / n) * sum_i (y_true_i - y_pred_i)^2

    Parametry
    ---------
    y_true : np.ndarray
        Skutecne hodnoty, tvar `(n,)`.
    y_pred : np.ndarray
        Predikce, stejny tvar jako `y_true`.

    Navratova hodnota
    -----------------
    float
        Hodnota MSE (python `float`), nezaporna.
    """
    # assert  Ověřte, že y_true a y_pred maji stejny tvar
    raise NotImplementedError(
        "Úkol: vratte prumer druhych mocnin rozdilu y_true a y_pred jako python float."
    )


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Vypocte stredni absolutni chybu, Mean Absolute Error (UKOL).

    Definice
    --------
    MAE = (1 / n) * sum_i |y_true_i - y_pred_i|

    Na rozdil od MSE je v jednotkach dat (u teploty ve stupnich) a odlehle
    hodnoty tresta jen linearne.

    Parametry
    ---------
    y_true : np.ndarray
        Skutecne hodnoty, tvar `(n,)`.
    y_pred : np.ndarray
        Predikce, stejny tvar jako `y_true`.

    Navratova hodnota
    -----------------
    float
        Hodnota MAE (python `float`), nezaporna.
    """
    # assert  Ověřte, že y_true a y_pred maji stejny tvar
    raise NotImplementedError(
        "Úkol: vratte prumer absolutnich hodnot rozdilu y_true a y_pred jako python float."
    )


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Vypocte koeficient determinace R^2, podil vysvetleneho rozptylu (UKOL).

    Definice
    --------
    SS_res = sum_i (y_true_i - y_pred_i)^2       (soucet ctvercu chyb modelu)
    SS_tot = sum_i (y_true_i - mean(y_true))^2   (celkovy soucet ctvercu odchylek
                                                  od prumeru, n-nasobek rozptylu)
    R^2    = 1 - SS_res / SS_tot

    Pozor: `SS_tot` se pocita z prumeru **skutecnych** hodnot `y_true`,
    ne z prumeru predikci.

    Vyznam: `R^2 = 1` - model vysvetlil vsechen rozptyl; `R^2 = 0` - model je
    stejne dobry jako predikce prumerem; `R^2 < 0` - model je horsi nez prumer.

    Parametry
    ---------
    y_true : np.ndarray
        Skutecne hodnoty, tvar `(n,)`.
    y_pred : np.ndarray
        Predikce, stejny tvar jako `y_true`.

    Navratova hodnota
    -----------------
    float
        Hodnota R^2 (python `float`), nejvyse 1, muze byt zaporna.
    """
    # assert  Ověřte, že y_true a y_pred maji stejny tvar
    # assert  Ověřte, že ss_tot > 0 (y_true neni konstantni)
    raise NotImplementedError(
        "Úkol: spocitejte ss_res a ss_tot podle docstringu a vratte 1 - ss_res / ss_tot "
        "jako python float."
    )


def persistence_baseline(series: np.ndarray, window_size: int) -> np.ndarray:
    """Naivni predpoved "zitra bude stejne jako dnes" zarovnana s cili oken (UKOL).

    Okno `i` datasetu `WindowedDataset(series, window_size)` ma vstup
    `series[i : i + window_size]` a cil `series[i + window_size]`. Persistence
    predpovida kazdy cil jako **posledni znamou hodnotu pred nim**, tedy
    hodnotu, kterou ma okno na svem poslednim miste.

    Vystup ma tvar `(len(series) - window_size,)` a jeho prvek `i` patri
    cili `series[window_size + i]`. Baseline musi byt zarovnana s cili presne.

    Pozor na chybu o jedna. Dve typicke spatne varianty:

    - vratit samotne cile (posun o nula): "predpoved" by znala budoucnost
      a chyba by vysla nulova,
    - posunout se o dva dny zpet misto o jeden: baseline je pak zbytecne
      horsi, nez by byla, a sit ji porazi "zadarmo".

    Priklad
    -------
    series = [10, 11, 12, 13, 14], window_size = 3
    cile                          = [13, 14]
    baseline                      = [12, 13]
    (okno [10, 11, 12] -> predpoved 12, cil 13; okno [11, 12, 13] -> 13, cil 14)

    Parametry
    ---------
    series : np.ndarray
        Casova rada, 1D pole tvaru `(n,)`.
    window_size : int
        Delka okna, `1 <= window_size < n`.

    Navratova hodnota
    -----------------
    np.ndarray
        Pole float64 tvaru `(n - window_size,)`; nesdili pamet se vstupem.
    """
    # assert  Ověřte, že series je 1D pole
    # assert  Ověřte, že window_size >= 1
    # assert  Ověřte, že len(series) > window_size
    raise NotImplementedError(
        "Úkol: vratte pole predpovedi, kde predpoved pro kazdy cil je posledni "
        "znama hodnota pred nim (viz priklad v docstringu); vysledek musi byt nove pole "
        "(ne pohled na vstup) tvaru (len(series) - window_size,)."
    )
