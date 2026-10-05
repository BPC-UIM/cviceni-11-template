"""Model MLP (viceurovnovy perceptron) v PyTorch.

Prevodni mustek cv10 -> cv11
----------------------------
Vsechno, co jste v cv10 psali rucne v NumPy, ma v PyTorch hotovou obdobu:

- `Neuron(Linear, Activation)`  ->  `nn.Linear` + `nn.Tanh` (dva samostatne moduly),
- `Sequential` (vlastni kontejner vrstev)  ->  `nn.Sequential`,
- inicializaci vah (`WeightsInitializer`) dela PyTorch sam pri vytvoreni `nn.Linear`,
- `backward` pisat nemusime: gradienty spocita `autograd` (viz `Trainer._train_step`).

Vystup je LINEARNI: posledni vrstva je `nn.Linear` bez aktivace. Ulohou je regrese,
tedy predikce spojite hodnoty (teplota, mira progrese onemocneni), ne pravdepodobnosti.
Sigmoida ci tanh na vystupu by omezila obor hodnot (napr. na (-1, 1)) a sit by nemohla
predpovidat libovolne cislo.

Model je sdileny pro obe casti cviceni (regrese na diabetes datech i predpoved
teploty); lisi se jen `input_dim` (10 priznaku, resp. delka okna).

POZNAMKA: sit je zamerne mala. Stara verze cviceni prokladala 1D sinusovku siti
6x70 neuronu, coz je zbytecne hluboke a spatne se uci. Pro nase ulohy staci
jedna az dve skryte vrstvy po nekolika desitkach neuronu.
"""

from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    """Plne propojena sit: `Linear -> Tanh` pro kazdou skrytou vrstvu (UKOL: __init__ a forward).

    Atributy
    --------
    input_dim : int
        Pocet vstupnich priznaku.
    hidden_dims : list[int]
        Pocty neuronu ve skrytych vrstvach.
    output_dim : int
        Pocet vystupu (pro regresi 1).
    net : nn.Sequential
        Vlastni sit; posledni modul je `nn.Linear` bez aktivace.
    """

    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int = 1) -> None:
        """Vytvori vrstvy site (UKOL).

        Parametry
        ---------
        input_dim : int
            Pocet vstupnich priznaku.
        hidden_dims : list[int]
            Pocty neuronu ve skrytych vrstvach, napr. `[8, 4]`.
        output_dim : int, optional
            Pocet vystupu (vychozi 1).

        Definice
        --------
        Pro vstup `(B, input_dim)` je vystup `(B, output_dim)`. Za kazdou skrytou
        vrstvou je aktivace `nn.Tanh`, za poslednim `nn.Linear` zadna aktivace.
        Vrstvy slozte do `nn.Sequential` a ulozte do atributu `self.net`.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        self.output_dim = output_dim

        # assert  Ověřte, že input_dim a output_dim jsou alespon 1
        # assert  Ověřte, že hidden_dims je neprazdny seznam kladnych cisel
        raise NotImplementedError(
            "Úkol: sestavte self.net jako nn.Sequential: pro kazdou skrytou vrstvu "
            "nn.Linear (pocet vstupu = vystup predchozi vrstvy) a za nim nn.Tanh, "
            "nakonec nn.Linear na output_dim BEZ aktivace (linearni vystup)."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Dopredny pruchod siti (UKOL).

        Parametry
        ---------
        x : torch.Tensor
            Vstupni davka, tvar `(B, input_dim)`, `float32`.

        Navratova hodnota
        -----------------
        torch.Tensor
            Predikce, tvar `(B, output_dim)`.
        """
        # assert  Ověřte, že x ma 2 rozmery a x.shape[1] == input_dim
        raise NotImplementedError(
            "Úkol: vratte vystup self.net pro vstup x (nic dalsiho nepocitejte)."
        )
