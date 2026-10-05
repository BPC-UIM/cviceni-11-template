"""Datasety: jak PyTorch cte data (protokol `Dataset` a navazujici `DataLoader`).

`torch.utils.data.Dataset` je jednoduchy protokol, tvoreny dvema metodami:

- `__len__` - kolik ma dataset vzorku,
- `__getitem__(idx)` - vrati `idx`-ty vzorek (zde dvojici `(x, y)`).

Dataset nic jineho nedela: nemicha, nedeli na davky a nedeli na trenovaci
a testovaci cast. Rozdeleni dat se dela predem (`dataio.random_split`,
`dataio.chronological_split`) a pro kazdou cast se vytvori vlastni dataset.

Na dataset navazuje `torch.utils.data.DataLoader`. Ten si od datasetu
postupne vyzada vzorky, slozi je do davek (`x_batch` tvaru `(B, n_features)`,
`y_batch` tvaru `(B, 1)`) a pripadne je pred kazdou epochou zamicha
(`shuffle=True`). Tim nahrazuje rucni davkovani z cv10, kde to delal
predvyplneny `Trainer.run` rucne pres `rng.permutation` a rezy pole.

Konvence tvaru (zafixovana pro cele cviceni): vsechny tenzory jsou
`torch.float32`; jeden vzorek je `x` tvaru `(n_features,)` a `y` tvaru `(1,)`.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class TabularDataset(Dataset):
    """Tabulkova data: jeden radek tabulky je jeden vzorek (PREDVYPLNENO, VZOR).

    Je to nejjednodussi mozny dataset a slouzi jako vzor pro `WindowedDataset`.
    Vsechna data se v `__init__` jednou prevedou na tenzory a `__getitem__`
    pak jen vybere radek.

    Priklad
    -------
    x tvaru `(6, 2)` a y tvaru `(6,)` -> `len(ds) == 6`, `ds[0]` je dvojice
    tenzoru tvaru `(2,)` a `(1,)`.

    Atributy
    --------
    x : torch.Tensor
        Priznaky, tvar `(n, d)`, `float32`.
    y : torch.Tensor
        Cile, tvar `(n, 1)`, `float32`.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        """Ulozi priznaky a cile jako `float32` tenzory (PREDVYPLNENO).

        Parametry
        ---------
        x : np.ndarray
            Priznaky, tvar `(n, d)`; tvar `(n,)` se prevede na `(n, 1)`.
        y : np.ndarray
            Cile, tvar `(n,)` nebo `(n, 1)`; vzdy se prevede na `(n, 1)`.
        """
        x_arr = np.asarray(x, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)
        # Jednorozmerny vstup je jeden priznak: (n,) -> (n, 1).
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(-1, 1)
        # Cil musi byt (n,) nebo (n, 1); vice vystupu by reshape ticho zplostil.
        assert y_arr.ndim == 1 or (y_arr.ndim == 2 and y_arr.shape[1] == 1), (
            f"y musi mit tvar (n,) nebo (n, 1), ne {y_arr.shape}"
        )
        # Cil vzdy jako sloupec, aby davka mela tvar (B, 1) shodny s vystupem modelu.
        y_arr = y_arr.reshape(-1, 1)
        assert x_arr.ndim == 2, "x musi mit tvar (n, d) nebo (n,)"
        assert len(x_arr) == len(y_arr), "x a y musi mit stejny pocet vzorku"
        # as_tensor sdili pamet s NumPy polem, pokud to jde (zadne zbytecne kopirovani).
        self.x = torch.as_tensor(x_arr)
        self.y = torch.as_tensor(y_arr)

    def __len__(self) -> int:
        """Vrati pocet vzorku (radku tabulky) (PREDVYPLNENO).

        Navratova hodnota
        -----------------
        int
            `n`, tj. pocet radku `x`.
        """
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Vrati `idx`-ty vzorek (PREDVYPLNENO).

        Parametry
        ---------
        idx : int
            Index vzorku, `0 <= idx < len(self)`.

        Navratova hodnota
        -----------------
        tuple[torch.Tensor, torch.Tensor]
            Dvojice `(x, y)` s tvary `(d,)` a `(1,)`, oba `float32`.
        """
        return self.x[idx], self.y[idx]


class WindowedDataset(Dataset):
    """Okenkovy dataset pro predpoved casove rady (UKOL).

    Z jedne rady hodnot `series` (napr. denni minimalni teploty) vyrobi
    vzorky typu "z poslednich `window_size` hodnot predpovez tu dalsi".

    DULEZITE: okno je nad HODNOTAMI SIGNALU, ne nad casovou osou. Vstup `x`
    jsou minule hodnoty rady, ne indexy ani cas. Stara verze cviceni
    okenkovala casovou osu (model pak dostaval cas, ne minulost signalu);
    tohle je oprava.

    Definice
    --------
    x_i = series[i : i + window_size]        tvar `(window_size,)`
    y_i = series[i + window_size]            tvar `(1,)`
    pocet vzorku = len(series) - window_size

    Maly priklad
    ------------
    series = [10, 11, 12, 13, 14], window_size = 3
    -> ([10, 11, 12], 13), ([11, 12, 13], 14)     (dva vzorky, 5 - 3 = 2)

    Atributy
    --------
    series : torch.Tensor
        Rada hodnot, tvar `(n,)`, `float32`.
    window_size : int
        Delka vstupniho okna (pocet minulych hodnot).

    Poznamka: dataset nemicha a nedeli. Rozdeleni rady na trenovaci
    a testovaci cast dela `dataio.chronological_split` pred vytvorenim datasetu.
    """

    def __init__(self, series: np.ndarray, window_size: int) -> None:
        """Ulozi radu jako 1D `float32` tenzor a delku okna (UKOL).

        Parametry
        ---------
        series : np.ndarray
            Rada hodnot signalu, tvar `(n,)`.
        window_size : int
            Delka vstupniho okna, `window_size >= 1`.
        """
        # assert  Ověřte, že series je 1D pole (rozmer zkontrolujte pred jakymkoli reshape)
        # assert  Ověřte, že window_size >= 1
        # assert  Ověřte, že len(series) > window_size
        raise NotImplementedError(
            "Úkol: prevedte radu na 1D pole float32 a z nej na tenzor; "
            "ulozte self.series a self.window_size."
        )

    def __len__(self) -> int:
        """Vrati pocet oken (vzorku) (UKOL).

        Navratova hodnota
        -----------------
        int
            `len(self.series) - self.window_size`.
        """
        raise NotImplementedError(
            "Úkol: vratte pocet oken = delka rady minus window_size."
        )

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Vrati `idx`-te okno a hodnotu, ktera po nem nasleduje (UKOL).

        Index mimo rozsah konci `IndexError` (z indexovani tenzoru), diky cemuz
        funguje i prima iterace pres dataset (`for x, y in ds`).

        Parametry
        ---------
        idx : int
            Index okna, `0 <= idx < len(self)`.

        Navratova hodnota
        -----------------
        tuple[torch.Tensor, torch.Tensor]
            Dvojice `(x, y)` s tvary `(window_size,)` a `(1,)`, oba `float32`.
        """
        raise NotImplementedError(
            "Úkol: x = window_size po sobe jdoucich hodnot signalu od indexu idx, "
            "y = hodnota hned za oknem jako tenzor tvaru (1,); vratte dvojici (x, y)."
        )
