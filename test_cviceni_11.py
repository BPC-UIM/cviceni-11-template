# -*- coding: utf-8 -*-

"""
Created on 02. 10. 2026 at 14:20:00

Author: Richard Redina
Email: 195715@vut.cz
Affiliation:
         International Clinical Research Center, Brno
         Brno University of Technology, Brno
GitHub: RicRedi

(._.)
 <|>
_/|_

Description:
    Testy pro cviceni 11 -- regrese a predpoved neuronovou siti v PyTorch.

    Spousteni:  pytest -v

    Ve stavu stubu se sada NACTE a testy, ktere volaji nedokoncene ukoly,
    se oznaci jako xfail (ocekavane selhani s NotImplementedError) -- sada
    nikdy neskonci holym tracebackem. Testy predvyplnenych casti
    (TabularDataset, ResidualAnalyzer, dataio) projdou v kazdem stavu.

    Testy nikdy nezaklada soubory v logs/ (logger se hned po importech
    nakonfiguruje bez souboru) a modely i grafy ukladaji jen do pytest
    ``tmp_path``. Testy TestTrainStep, TestTrainer a TestSaveLoad pouzivaji
    jako model samotnou vrstvu ``torch.nn.Linear``, takze nezavisi na
    studentove ``MLP``.

    Trinact trid testu:
      TestMLP                  -- tvar vystupu, vrstvy Linear, Tanh, linearni vystup
      TestTrainStep            -- jeden krok uceni (jadro cviceni)
      TestTrainer              -- predvyplnene fit a predict
      TestSaveLoad             -- state_dict do souboru a zpet
      TestTabularDataset       -- hotovy vzor datasetu
      TestWindowedDataset      -- okno nad hodnotami signalu, ne nad casovou osou
      TestMetriky              -- mse, mae, r2 proti sklearn.metrics
      TestPersistenceBaseline  -- tvar a zarovnani s cili oken
      TestResidualAnalyzer     -- predvyplnena rezidualni diagnostika
      TestDataAKonfigurace     -- loadery, rozdeleni dat, konfigurace (vc. nove sekce)
      TestPipelineUtils        -- utils/: kontrola uceni, hlasky, predikce (bez studenta)
      TestUceniEndToEnd        -- MLP + Trainer na jednoduche zavislosti
      TestForecastNaTeplotach  -- skutecna rada teplot (preskoci se bez CSV)
================================================================================
"""

from __future__ import annotations

import copy
import dataclasses
import os
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dataio import (
    TEMPERATURE_CSV,
    DiagnosticsConfig,
    ExperimentConfig,
    TrainingConfig,
    chronological_split,
    load_config,
    load_diabetes_data,
    load_temperature_series,
    make_sinusoid,
    random_split,
    validate_config,
)
from src import (
    MLP,
    ResidualAnalyzer,
    TabularDataset,
    Trainer,
    WindowedDataset,
    mae,
    mse,
    persistence_baseline,
    r2,
)
from src.trainer import get_logger
from utils import UceniDivergovalo, chyba_dat, kontrola_uceni, predikuj, preskoceno

get_logger(None)   # logger jen do konzole: testy nikdy nezalozi soubor v logs/

STUB = pytest.mark.xfail(raises=NotImplementedError, strict=False,
                         reason="studentsky ukol jeste neni dokoncen")

ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_YAML = os.path.join(ROOT, "config.yaml")
TEMPERATURE_PATH = os.path.join(ROOT, TEMPERATURE_CSV)


# --------------------------------------------------------------------------- #
#  Pomocne funkce                                                             #
# --------------------------------------------------------------------------- #
def _cfg(lr: float = 0.1, epochs: int = 3, batch_size: int = 8) -> TrainingConfig:
    """Vytvori ``TrainingConfig`` primo (bez config.yaml)."""
    return TrainingConfig(lr=lr, epochs=epochs, batch_size=batch_size, seed=0)


def _linear_data(n: int = 32, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Davka dat ``y = 2x + 1``: ``x`` tvaru ``(n, 1)``, ``y`` tvaru ``(n, 1)``."""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 1, generator=g) * 2.0 - 1.0
    return x, 2.0 * x + 1.0


def _grad_vector(model: nn.Module) -> torch.Tensor:
    """Slepi gradienty vsech parametru do jednoho vektoru (kopie)."""
    for p in model.parameters():
        assert p.grad is not None, (
            "po _train_step musi zustat gradient z posledniho backward; "
            "volejte zero_grad PRED backward"
        )
    return torch.cat([p.grad.detach().clone().reshape(-1) for p in model.parameters()])


# --------------------------------------------------------------------------- #
#  MLP                                                                        #
# --------------------------------------------------------------------------- #
class TestMLP:
    """Model ``MLP`` -- struktura site a linearni vystup."""

    @STUB
    def test_tvar_vystupu(self) -> None:
        """Vystup ma tvar ``(B, output_dim)``."""
        model = MLP(5, [8, 4])
        assert model(torch.randn(7, 5)).shape == (7, 1)
        assert MLP(5, [8], output_dim=3)(torch.randn(7, 5)).shape == (7, 3)

    @STUB
    def test_pocet_a_rozmery_linear(self) -> None:
        """Pocet ``nn.Linear`` je ``len(hidden_dims) + 1`` a rozmery na sebe navazuji."""
        hidden = [8, 4, 2]
        model = MLP(5, hidden)
        linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
        assert len(linears) == len(hidden) + 1
        vstupy = [5] + hidden
        vystupy = hidden + [1]
        for lin, n_in, n_out in zip(linears, vstupy, vystupy):
            assert (lin.in_features, lin.out_features) == (n_in, n_out)

    @STUB
    def test_tanh_za_kazdou_skrytou_vrstvou(self) -> None:
        """Za kazdou skrytou ``nn.Linear`` nasleduje ``nn.Tanh``."""
        model = MLP(5, [8, 4])
        moduly = list(model.net)
        pocet_tanh = sum(isinstance(m, nn.Tanh) for m in moduly)
        assert pocet_tanh == 2
        for k, m in enumerate(moduly[:-1]):
            if isinstance(m, nn.Linear):
                assert isinstance(moduly[k + 1], nn.Tanh)

    @STUB
    def test_posledni_vrstva_je_linearni(self) -> None:
        """Posledni modul ``self.net`` je ``nn.Linear`` (bez aktivace)."""
        model = MLP(5, [8, 4])
        assert isinstance(list(model.net)[-1], nn.Linear)

    @STUB
    def test_vystup_muze_byt_mimo_interval_minus1_az_1(self) -> None:
        """Linearni vystup neni omezen na ``(-1, 1)`` (jako by byl u tanh na konci)."""
        torch.manual_seed(0)
        model = MLP(1, [4])
        with torch.no_grad():
            last = list(model.net)[-1]
            last.weight.fill_(10.0)
            last.bias.fill_(5.0)
            y = model(torch.linspace(-3, 3, 20).reshape(-1, 1))
        assert float(y.max()) > 1.0 or float(y.min()) < -1.0

    @STUB
    def test_model_ma_parametry(self) -> None:
        """Model ma parametry (vrstvy jsou zaregistrovane v ``nn.Module``)."""
        model = MLP(3, [4])
        pocet = sum(p.numel() for p in model.parameters())
        assert pocet == 3 * 4 + 4 + 4 * 1 + 1


# --------------------------------------------------------------------------- #
#  Trainer._train_step                                                        #
# --------------------------------------------------------------------------- #
class TestTrainStep:
    """Jadro cviceni: ``Trainer._train_step`` (model je samotna ``nn.Linear``)."""

    @STUB
    def test_vraci_float(self) -> None:
        """Krok vraci Python ``float``."""
        trainer = Trainer(nn.Linear(1, 1), _cfg(), log_dir=None)
        x, y = _linear_data()
        assert isinstance(trainer._train_step(x, y), float)

    @STUB
    def test_vracena_ztrata_je_mse_pred_krokem(self) -> None:
        """Vracena ztrata je MSE spocitane PRED aktualizaci vah (rucne pres ``no_grad``)."""
        model = nn.Linear(1, 1)
        trainer = Trainer(model, _cfg(lr=0.1), log_dir=None)
        x, y = _linear_data()
        with torch.no_grad():
            ocekavana = float(((model(x) - y) ** 2).mean())
        assert trainer._train_step(x, y) == pytest.approx(ocekavana, rel=1e-5)

    @STUB
    def test_vahy_po_kroku_jsou_w_minus_lr_krat_gradient(self) -> None:
        """Po jednom kroku plati ``w_new == w_old - lr * grad`` (chyti dvojity ``step``)."""
        lr = 0.1
        model = nn.Linear(1, 1)
        kopie = copy.deepcopy(model)
        x, y = _linear_data()
        ztrata = ((kopie(x) - y) ** 2).mean()          # gradient spocitany rucne na kopii
        ztrata.backward()
        ocekavane = [p.detach() - lr * p.grad for p in kopie.parameters()]

        Trainer(model, _cfg(lr=lr), log_dir=None)._train_step(x, y)
        for p, e in zip(model.parameters(), ocekavane):
            assert torch.allclose(p.detach(), e, atol=1e-6)

    @STUB
    def test_meni_vahy(self) -> None:
        """Po kroku se vahy modelu zmeni."""
        model = nn.Linear(1, 1)
        trainer = Trainer(model, _cfg(), log_dir=None)
        pred = [p.detach().clone() for p in model.parameters()]
        x, y = _linear_data()
        trainer._train_step(x, y)
        zmena = sum(float((a - b.detach()).abs().sum()) for a, b in zip(pred, model.parameters()))
        assert zmena > 0.0

    @STUB
    def test_opakovane_kroky_snizuji_ztratu(self) -> None:
        """Opakovane kroky na ``y = 2x + 1`` snizi ztratu."""
        torch.manual_seed(0)
        trainer = Trainer(nn.Linear(1, 1), _cfg(lr=0.2), log_dir=None)
        x, y = _linear_data()
        prvni = trainer._train_step(x, y)
        for _ in range(100):
            posledni = trainer._train_step(x, y)
        assert posledni < 0.1 * prvni

    @STUB
    def test_gradienty_se_nescitaji(self) -> None:
        """S ``lr = 0`` je ``.grad`` po druhem kroku na teze davce stejny jako po prvnim."""
        model = nn.Linear(1, 1)
        trainer = Trainer(model, _cfg(lr=0.0), log_dir=None)
        x, y = _linear_data()
        trainer._train_step(x, y)
        po_prvnim = _grad_vector(model)
        trainer._train_step(x, y)
        po_druhem = _grad_vector(model)
        assert torch.any(po_prvnim != 0)
        assert torch.allclose(po_prvnim, po_druhem), "gradienty se scitaji -- chybi zero_grad()"

    @STUB
    def test_nesouhlas_tvaru_je_chyba(self) -> None:
        """``y_batch`` tvaru ``(B,)`` proti vystupu ``(B, 1)`` vyhodi vyjimku."""
        trainer = Trainer(nn.Linear(1, 1), _cfg(), log_dir=None)
        x, y = _linear_data(n=6)
        with pytest.raises((AssertionError, ValueError)):
            trainer._train_step(x, y.reshape(-1))


# --------------------------------------------------------------------------- #
#  Trainer: fit a predict                                                     #
# --------------------------------------------------------------------------- #
class TestTrainer:
    """Predvyplnene ``fit`` a ``predict`` (model je samotna ``nn.Linear``)."""

    @staticmethod
    def _loadery(batch_size: int = 8) -> tuple[DataLoader, DataLoader]:
        """Trenovaci a validacni loader nad ``y = 2x + 1``."""
        x, y = _linear_data(n=30)
        train = TabularDataset(x.numpy(), y.numpy())
        val = TabularDataset(x.numpy()[:10], y.numpy()[:10])
        return (DataLoader(train, batch_size=batch_size, shuffle=True),
                DataLoader(val, batch_size=batch_size, shuffle=False))

    @STUB
    def test_klice_a_delky_historie(self) -> None:
        """Historie ma klice ``epoch``, ``train_loss``, ``val_loss`` o delce ``epochs``."""
        trainer = Trainer(nn.Linear(1, 1), _cfg(epochs=4), log_dir=None)
        history = trainer.fit(*self._loadery())
        assert set(history) == {"epoch", "train_loss", "val_loss"}
        assert history["epoch"] == [1, 2, 3, 4]
        assert len(history["train_loss"]) == len(history["val_loss"]) == 4
        assert trainer.history_ is history

    @STUB
    def test_model_po_fit_je_v_eval(self) -> None:
        """Po ``fit`` je model ve stavu ``eval`` (``model.training`` je ``False``)."""
        model = nn.Linear(1, 1)
        trainer = Trainer(model, _cfg(epochs=2), log_dir=None)
        trainer.fit(*self._loadery())
        assert model.training is False

    @STUB
    def test_fit_snizuje_ztratu(self) -> None:
        """Trenovaci ztrata po ``fit`` je nizsi nez po prvni epose."""
        trainer = Trainer(nn.Linear(1, 1), _cfg(lr=0.1, epochs=30), log_dir=None)
        history = trainer.fit(*self._loadery())
        assert history["train_loss"][-1] < history["train_loss"][0]

    def test_predict_nestavi_graf(self) -> None:
        """``predict`` vraci tenzor s ``requires_grad`` ``False`` a tvarem ``(n, 1)``."""
        trainer = Trainer(nn.Linear(2, 1), _cfg(), log_dir=None)
        y = trainer.predict(torch.randn(5, 2))
        assert isinstance(y, torch.Tensor)
        assert y.requires_grad is False
        assert y.shape == (5, 1)

    def test_evaluate_s_cili_spatneho_tvaru_je_value_error(self) -> None:
        """``_evaluate`` s cili tvaru ``(B,)`` proti vystupu ``(B, 1)`` vyhodi ``ValueError``."""
        trainer = Trainer(nn.Linear(1, 1), _cfg(), log_dir=None)
        x, y = _linear_data(n=8)
        loader = DataLoader(TensorDataset(x, y.reshape(-1)), batch_size=4)
        with pytest.raises(ValueError):
            trainer._evaluate(loader)

    def test_evaluate_prazdny_loader_je_value_error(self) -> None:
        """``_evaluate`` s prazdnym loaderem vyhodi ``ValueError``."""
        trainer = Trainer(nn.Linear(1, 1), _cfg(), log_dir=None)
        prazdny = DataLoader(TabularDataset(np.zeros((0, 1)), np.zeros(0)), batch_size=4)
        with pytest.raises(ValueError):
            trainer._evaluate(prazdny)

    def test_fit_prazdny_trenovaci_loader_je_value_error(self) -> None:
        """``fit`` s prazdnym trenovacim loaderem vyhodi ``ValueError``."""
        trainer = Trainer(nn.Linear(1, 1), _cfg(), log_dir=None)
        _, val = self._loadery()
        prazdny = DataLoader(TabularDataset(np.zeros((0, 1)), np.zeros(0)), batch_size=4)
        with pytest.raises(ValueError):
            trainer.fit(prazdny, val)

    def test_fit_prazdny_validacni_loader_je_value_error(self) -> None:
        """``fit`` s prazdnym validacnim loaderem vyhodi ``ValueError``."""

        class _Trainer(Trainer):
            """Trainer s nahradnim krokem, aby test nezavisel na studentovi."""

            def _train_step(self, x_batch: torch.Tensor, y_batch: torch.Tensor) -> float:
                """Vraci pevnou ztratu jako ``float``."""
                return 0.0

        train, _ = self._loadery()
        prazdny = DataLoader(TabularDataset(np.zeros((0, 1)), np.zeros(0)), batch_size=4)
        with pytest.raises(ValueError):
            _Trainer(nn.Linear(1, 1), _cfg(), log_dir=None).fit(train, prazdny)

    def test_fit_train_step_vraci_tenzor_je_type_error(self) -> None:
        """``fit`` vyhodi ``TypeError``, kdyz ``_train_step`` vrati tenzor misto ``float``."""

        class _Trainer(Trainer):
            """Trainer, jehoz krok vraci tenzor (typicka chyba: chybi ``.item()``)."""

            def _train_step(self, x_batch: torch.Tensor, y_batch: torch.Tensor) -> float:
                """Vraci tenzor misto ``float``."""
                return torch.tensor(0.0)

        with pytest.raises(TypeError):
            _Trainer(nn.Linear(1, 1), _cfg(), log_dir=None).fit(*self._loadery())

    def test_predict_prepne_model_do_eval(self) -> None:
        """``predict`` nechava model ve stavu ``eval``."""
        model = nn.Linear(2, 1)
        model.train()
        Trainer(model, _cfg(), log_dir=None).predict(torch.randn(3, 2))
        assert model.training is False


# --------------------------------------------------------------------------- #
#  Trainer: save a load                                                       #
# --------------------------------------------------------------------------- #
class TestSaveLoad:
    """``Trainer.save`` a ``Trainer.load`` (model je samotna ``nn.Linear``)."""

    @STUB
    def test_round_trip_do_nove_instance(self, tmp_path: Path) -> None:
        """Nova instance po ``load`` dava shodne predikce jako puvodni model."""
        torch.manual_seed(1)
        puvodni = Trainer(nn.Linear(3, 1), _cfg(), log_dir=None)
        cesta = str(tmp_path / "model.pt")
        puvodni.save(cesta)

        torch.manual_seed(2)
        nova = nn.Linear(3, 1)
        x = torch.randn(10, 3)
        assert not torch.allclose(nova(x), puvodni.model(x)), "nova sit musi zacit jinak"
        nacteny = Trainer.load(cesta, nova, _cfg())
        assert isinstance(nacteny, Trainer)
        assert torch.allclose(nacteny.predict(x), puvodni.predict(x))

    @STUB
    def test_soubor_je_state_dict(self, tmp_path: Path) -> None:
        """Soubor obsahuje slovnik se stejnymi klici jako ``model.state_dict()``."""
        model = nn.Linear(3, 1)
        cesta = str(tmp_path / "model.pt")
        Trainer(model, _cfg(), log_dir=None).save(cesta)
        obsah = torch.load(cesta, weights_only=True)
        assert isinstance(obsah, dict)
        assert set(obsah) == set(model.state_dict())
        for klic, tenzor in model.state_dict().items():
            assert torch.equal(obsah[klic], tenzor)

    @STUB
    def test_load_neexistujiciho_souboru_je_chyba(self, tmp_path: Path) -> None:
        """``load`` neexistujiciho souboru vyhodi ``AssertionError`` nebo ``FileNotFoundError``."""
        with pytest.raises((AssertionError, FileNotFoundError)):
            Trainer.load(str(tmp_path / "neexistuje.pt"), nn.Linear(2, 1), _cfg())

    @STUB
    def test_soubor_nebyl_ulozen_jako_cely_objekt(self, tmp_path: Path) -> None:
        """Soubor nelze nacist jako cely model (neni to pickle objektu)."""
        cesta = str(tmp_path / "model.pt")
        Trainer(nn.Linear(2, 1), _cfg(), log_dir=None).save(cesta)
        obsah = torch.load(cesta, weights_only=True)
        assert not isinstance(obsah, nn.Module)


# --------------------------------------------------------------------------- #
#  TabularDataset                                                             #
# --------------------------------------------------------------------------- #
class TestTabularDataset:
    """Hotovy vzor ``TabularDataset``."""

    def test_delka_a_tvary(self) -> None:
        """Delka je pocet vzorku, ``x`` ma tvar ``(d,)`` a ``y`` tvar ``(1,)``."""
        ds = TabularDataset(np.random.default_rng(0).normal(size=(12, 4)), np.arange(12.0))
        assert len(ds) == 12
        x, y = ds[3]
        assert x.shape == (4,)
        assert y.shape == (1,)

    def test_typ_float32(self) -> None:
        """Vzorky jsou ``torch.float32`` i pri vstupu ``float64``."""
        ds = TabularDataset(np.zeros((5, 2), dtype=np.float64), np.zeros(5, dtype=np.float64))
        x, y = ds[0]
        assert x.dtype == torch.float32
        assert y.dtype == torch.float32

    def test_1d_vstup_a_sloupcovy_cil(self) -> None:
        """``x`` tvaru ``(n,)`` se zmeni na ``(n, 1)`` a ``y`` tvaru ``(n, 1)`` projde."""
        ds = TabularDataset(np.arange(6.0), np.arange(6.0).reshape(-1, 1))
        x, y = ds[2]
        assert x.shape == (1,)
        assert y.shape == (1,)
        assert float(x[0]) == 2.0

    def test_hodnoty(self) -> None:
        """``__getitem__`` vraci odpovidajici radek dat."""
        x = np.arange(20.0).reshape(10, 2)
        y = np.arange(10.0) * 10.0
        xi, yi = TabularDataset(x, y)[4]
        assert xi.tolist() == [8.0, 9.0]
        assert float(yi[0]) == 40.0

    def test_davky_z_dataloaderu(self) -> None:
        """``DataLoader`` sklada davky ``(B, d)`` a ``(B, 1)``."""
        ds = TabularDataset(np.zeros((10, 3)), np.zeros(10))
        x_batch, y_batch = next(iter(DataLoader(ds, batch_size=4)))
        assert x_batch.shape == (4, 3)
        assert y_batch.shape == (4, 1)

    def test_ruzny_pocet_vzorku_je_chyba(self) -> None:
        """Ruzny pocet vzorku ``x`` a ``y`` vyhodi ``AssertionError``."""
        with pytest.raises(AssertionError):
            TabularDataset(np.zeros((5, 2)), np.zeros(4))


# --------------------------------------------------------------------------- #
#  WindowedDataset                                                            #
# --------------------------------------------------------------------------- #
class TestWindowedDataset:
    """``WindowedDataset`` -- okno nad hodnotami signalu, ne nad casovou osou."""

    # Hodnoty rady se LISI od indexu, aby test odhalil okenkovani casove osy.
    SERIE = 100.0 + 3.0 * np.arange(20)

    @STUB
    def test_delka(self) -> None:
        """Delka je ``len(series) - window_size``."""
        assert len(WindowedDataset(self.SERIE, 4)) == 16

    @STUB
    def test_x_jsou_minule_hodnoty_a_y_nasledujici(self) -> None:
        """``x == series[i : i + w]`` a ``y == series[i + w]`` pro vsechna ``i``."""
        w = 4
        ds = WindowedDataset(self.SERIE, w)
        for i in range(len(ds)):
            x, y = ds[i]
            assert x.tolist() == self.SERIE[i:i + w].tolist()
            assert float(y[0]) == self.SERIE[i + w]

    @STUB
    def test_nahodna_rada(self) -> None:
        """Totez na nahodne rade (float32 prevod, shoda na toleranci)."""
        serie = np.random.default_rng(3).normal(size=30)
        w = 5
        ds = WindowedDataset(serie, w)
        for i in range(len(ds)):
            x, y = ds[i]
            np.testing.assert_allclose(x.numpy(), serie[i:i + w], rtol=1e-6)
            np.testing.assert_allclose(y.numpy(), [serie[i + w]], rtol=1e-6)

    @STUB
    def test_prvni_a_posledni_okno(self) -> None:
        """Prvni okno zacina na zacatku rady, posledni konci na predposledni hodnote."""
        w = 3
        ds = WindowedDataset(self.SERIE, w)
        x0, y0 = ds[0]
        assert x0.tolist() == self.SERIE[:3].tolist()
        assert float(y0[0]) == self.SERIE[3]
        x_last, y_last = ds[len(ds) - 1]
        assert x_last.tolist() == self.SERIE[-4:-1].tolist()
        assert float(y_last[0]) == self.SERIE[-1]

    @STUB
    def test_tvary_a_typ(self) -> None:
        """``x`` ma tvar ``(w,)``, ``y`` tvar ``(1,)``, oba ``float32``."""
        x, y = WindowedDataset(self.SERIE, 6)[0]
        assert x.shape == (6,)
        assert y.shape == (1,)
        assert x.dtype == torch.float32
        assert y.dtype == torch.float32

    @STUB
    def test_davky_z_dataloaderu(self) -> None:
        """Davky z ``DataLoader`` maji tvary ``(B, w)`` a ``(B, 1)``."""
        ds = WindowedDataset(self.SERIE, 4)
        x_batch, y_batch = next(iter(DataLoader(ds, batch_size=5, shuffle=False)))
        assert x_batch.shape == (5, 4)
        assert y_batch.shape == (5, 1)
        assert x_batch[0].tolist() == self.SERIE[:4].tolist()

    @STUB
    def test_2d_rada_je_chyba(self) -> None:
        """2D rada (napr. tvaru ``(5, 2)``) vyhodi ``AssertionError`` nebo ``ValueError``."""
        with pytest.raises((AssertionError, ValueError)):
            WindowedDataset(np.arange(10.0).reshape(5, 2), 2)

    @STUB
    def test_iterace_konci_indexerrorem(self) -> None:
        """Prima iterace ``list(ds)`` ma delku ``len(ds)`` (index mimo rozsah da ``IndexError``)."""
        ds = WindowedDataset(self.SERIE, 4)
        assert len(list(ds)) == len(ds)
        with pytest.raises(IndexError):
            ds[len(ds)]

    @STUB
    def test_prilis_kratka_rada_je_chyba(self) -> None:
        """Rada kratsi nebo stejne dlouha jako okno vyhodi ``AssertionError``."""
        with pytest.raises(AssertionError):
            WindowedDataset(np.arange(4.0), 4)


# --------------------------------------------------------------------------- #
#  Metriky                                                                    #
# --------------------------------------------------------------------------- #
class TestMetriky:
    """``mse``, ``mae``, ``r2`` -- reference ``sklearn.metrics``."""

    @staticmethod
    def _data() -> tuple[np.ndarray, np.ndarray]:
        """Nahodna skutecnost a predikce tvaru ``(50,)``."""
        rng = np.random.default_rng(0)
        y_true = rng.normal(5.0, 2.0, size=50)
        return y_true, y_true + rng.normal(0.0, 1.0, size=50)

    @STUB
    def test_mse_proti_sklearn(self) -> None:
        """``mse`` odpovida ``sklearn.metrics.mean_squared_error``."""
        y_true, y_pred = self._data()
        assert mse(y_true, y_pred) == pytest.approx(mean_squared_error(y_true, y_pred))

    @STUB
    def test_mae_proti_sklearn(self) -> None:
        """``mae`` odpovida ``sklearn.metrics.mean_absolute_error``."""
        y_true, y_pred = self._data()
        assert mae(y_true, y_pred) == pytest.approx(mean_absolute_error(y_true, y_pred))

    @STUB
    def test_r2_proti_sklearn(self) -> None:
        """``r2`` odpovida ``sklearn.metrics.r2_score``."""
        y_true, y_pred = self._data()
        assert r2(y_true, y_pred) == pytest.approx(r2_score(y_true, y_pred))

    @STUB
    def test_r2_dokonala_predikce(self) -> None:
        """Dokonala predikce ma ``R^2 = 1``."""
        y = np.array([1.0, 2.0, 4.0, 8.0])
        assert r2(y, y.copy()) == pytest.approx(1.0)

    @STUB
    def test_r2_predikce_prumerem(self) -> None:
        """Predikce prumerem skutecnych hodnot ma ``R^2 = 0``."""
        y = np.array([1.0, 2.0, 4.0, 8.0, 10.0])
        assert r2(y, np.full_like(y, y.mean())) == pytest.approx(0.0, abs=1e-12)

    @STUB
    def test_r2_zaporne(self) -> None:
        """Predikce horsi nez prumer ma ``R^2 < 0``."""
        y = np.array([1.0, 2.0, 3.0, 4.0])
        assert r2(y, y[::-1] * 3.0) < 0.0

    @STUB
    def test_znama_hodnota(self) -> None:
        """Rucne spocitane hodnoty: chyby ``[1, -1, 2]``."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 1.0, 5.0])
        assert mse(y_true, y_pred) == pytest.approx(2.0)
        assert mae(y_true, y_pred) == pytest.approx(4.0 / 3.0)

    @STUB
    def test_navratovy_typ_je_float(self) -> None:
        """Vsechny tri metriky vraci Python ``float``."""
        y_true, y_pred = self._data()
        for metrika in (mse, mae, r2):
            assert isinstance(metrika(y_true, y_pred), float)

    @STUB
    def test_nesouhlas_tvaru_je_chyba(self) -> None:
        """``(n,)`` proti ``(n, 1)`` vyhodi ``AssertionError`` nebo ``ValueError``."""
        y_true, y_pred = self._data()
        for metrika in (mse, mae, r2):
            with pytest.raises((AssertionError, ValueError)):
                metrika(y_true, y_pred.reshape(-1, 1))


# --------------------------------------------------------------------------- #
#  persistence_baseline                                                       #
# --------------------------------------------------------------------------- #
class TestPersistenceBaseline:
    """``persistence_baseline`` -- naivni predpoved "zitra jako dnes"."""

    SERIE = np.random.default_rng(1).normal(10.0, 3.0, size=60)

    @STUB
    def test_tvar(self) -> None:
        """Vystup ma tvar ``(len(series) - window_size,)``."""
        assert persistence_baseline(self.SERIE, 7).shape == (len(self.SERIE) - 7,)

    @STUB
    def test_prvky_jsou_posledni_znama_hodnota(self) -> None:
        """``baseline[i] == series[w + i - 1]``."""
        w = 7
        baseline = persistence_baseline(self.SERIE, w)
        for i in range(len(baseline)):
            assert baseline[i] == pytest.approx(self.SERIE[w + i - 1])

    @STUB
    def test_ciselny_priklad(self) -> None:
        """Rada ``10..15`` s ``w = 3`` dava baseline ``[12, 13, 14]``."""
        rada = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
        assert persistence_baseline(rada, 3).tolist() == [12.0, 13.0, 14.0]

    @STUB
    def test_zarovnani_s_cili_windowed_datasetu(self) -> None:
        """``baseline[i]`` je posledni hodnota okna ``i`` a patri k cili okna ``i``."""
        w = 5
        ds = WindowedDataset(self.SERIE, w)
        baseline = persistence_baseline(self.SERIE, w)
        assert len(baseline) == len(ds)
        for i in range(len(ds)):
            x, y = ds[i]
            assert baseline[i] == pytest.approx(float(x[-1]), rel=1e-5)
            assert float(y[0]) == pytest.approx(self.SERIE[w + i], rel=1e-5)

    @STUB
    def test_chyba_je_nenulova(self) -> None:
        """MAE baseline na nahodne rade je ``> 0`` (baseline nejsou samotne cile)."""
        w = 5
        baseline = persistence_baseline(self.SERIE, w)
        assert mae(self.SERIE[w:], baseline) > 0.0

    @STUB
    def test_float64_a_nesdili_pamet_se_vstupem(self) -> None:
        """Vysledek je ``float64`` (i z ``float32`` vstupu) a nesdili pamet se vstupem."""
        baseline = persistence_baseline(self.SERIE, 5)
        assert baseline.dtype == np.float64
        assert not np.shares_memory(baseline, self.SERIE)
        assert persistence_baseline(self.SERIE.astype(np.float32), 5).dtype == np.float64

    @STUB
    def test_2d_rada_je_chyba(self) -> None:
        """2D rada (napr. tvaru ``(5, 2)``) vyhodi ``AssertionError`` nebo ``ValueError``."""
        with pytest.raises((AssertionError, ValueError)):
            persistence_baseline(np.arange(10.0).reshape(5, 2), 2)

    @STUB
    def test_neni_posunuta_o_jedna_zpet(self) -> None:
        """Baseline neni ``series[w - 2 : -2]`` (predpoved z predevcira)."""
        w = 5
        baseline = persistence_baseline(self.SERIE, w)
        assert not np.allclose(baseline, self.SERIE[w - 2:-2])


# --------------------------------------------------------------------------- #
#  ResidualAnalyzer                                                           #
# --------------------------------------------------------------------------- #
class TestResidualAnalyzer:
    """Predvyplnena diagnostika ``ResidualAnalyzer``."""

    @staticmethod
    def _analyzer(seed: int = 0) -> ResidualAnalyzer:
        """Analyzer nad nahodnymi daty a predikcemi."""
        rng = np.random.default_rng(seed)
        y_true = rng.normal(150.0, 50.0, size=120)
        return ResidualAnalyzer(y_true, y_true + rng.normal(0.0, 10.0, size=120))

    def test_rezidua_jsou_true_minus_pred(self) -> None:
        """Rezidua jsou ``y_true - y_pred`` tvaru ``(n,)``."""
        y_true = np.array([3.0, 5.0, 7.0])
        y_pred = np.array([2.0, 6.0, 7.0])
        analyzer = ResidualAnalyzer(y_true, y_pred)
        np.testing.assert_allclose(analyzer.compute_residuals(), [1.0, -1.0, 0.0])
        np.testing.assert_allclose(analyzer.residuals_, [1.0, -1.0, 0.0])

    def test_klice_normality(self) -> None:
        """Test normality vraci klice ``statistic``, ``p_value``, ``normal``."""
        vysledek = self._analyzer().residual_normality()
        assert set(vysledek) == {"statistic", "p_value", "normal"}
        assert 0.0 <= vysledek["p_value"] <= 1.0
        assert vysledek["normal"] == (vysledek["p_value"] > 0.05)

    def test_normalni_rezidua_vs_exponencialni(self) -> None:
        """Silne sikme rozdeleni je zamitnuto, rozdeleni normalni ne."""
        rng = np.random.default_rng(0)
        nuly = np.zeros(300)
        normalni = ResidualAnalyzer(nuly, rng.normal(size=300)).residual_normality()
        sikme = ResidualAnalyzer(nuly, rng.exponential(size=300)).residual_normality()
        assert normalni["normal"] is True
        assert sikme["normal"] is False

    def test_ruzna_delka_je_chyba(self) -> None:
        """Ruzna delka ``y_true`` a ``y_pred`` vyhodi ``ValueError``."""
        with pytest.raises(ValueError):
            ResidualAnalyzer(np.zeros(5), np.zeros(4))

    def test_nan_v_predikci_je_value_error(self) -> None:
        """NaN v predikcich (divergovane uceni) vyhodi ``ValueError``."""
        y_pred = np.array([1.0, 2.0, np.nan, 4.0])
        with pytest.raises(ValueError):
            ResidualAnalyzer(np.arange(4.0), y_pred)

    def test_mene_nez_tri_hodnoty_je_value_error(self) -> None:
        """Mene nez 3 hodnoty vyhodi ``ValueError``."""
        with pytest.raises(ValueError):
            ResidualAnalyzer(np.array([1.0, 2.0]), np.array([1.5, 2.5]))

    def test_grafy_se_ulozi(self, tmp_path: Path) -> None:
        """Vsechny ctyri grafy se ulozi jako neprazdne soubory do ``tmp_path``."""
        analyzer = self._analyzer()
        x = np.random.default_rng(1).normal(size=120)
        cesty = {
            "qq": tmp_path / "qq.png",
            "hist": tmp_path / "hist.png",
            "pred": tmp_path / "pred.png",
            "x": tmp_path / "x.png",
        }
        analyzer.qq_residuals(save_path=str(cesty["qq"]))
        analyzer.histogram_residuals(save_path=str(cesty["hist"]))
        analyzer.residuals_vs_prediction(save_path=str(cesty["pred"]))
        analyzer.residuals_vs_x(x, save_path=str(cesty["x"]), x_label="bmi")
        for cesta in cesty.values():
            assert cesta.is_file()
            assert cesta.stat().st_size > 0


# --------------------------------------------------------------------------- #
#  Data a konfigurace                                                         #
# --------------------------------------------------------------------------- #
class TestDataAKonfigurace:
    """Loadery, rozdeleni dat a konfigurace (predvyplneno)."""

    def test_diabetes_tvary(self) -> None:
        """Diabetes: 442 vzorku, 10 priznaku, rozdeleni 354 / 88."""
        x_tr, y_tr, x_te, y_te, _, _ = load_diabetes_data()
        assert x_tr.shape == (354, 10)
        assert x_te.shape == (88, 10)
        assert y_tr.shape == (354,)
        assert y_te.shape == (88,)

    def test_diabetes_standardizace_z_trenovaci_casti(self) -> None:
        """Trenovaci priznaky i cil maji prumer ~0 a odchylku ~1, ``y_mean`` v puv. jednotkach."""
        x_tr, y_tr, x_te, y_te, y_mean, y_std = load_diabetes_data()
        np.testing.assert_allclose(x_tr.mean(axis=0), 0.0, atol=1e-9)
        np.testing.assert_allclose(x_tr.std(axis=0), 1.0, atol=1e-9)
        assert y_tr.mean() == pytest.approx(0.0, abs=1e-9)
        assert y_tr.std() == pytest.approx(1.0, abs=1e-9)
        assert 100.0 < y_mean < 200.0
        assert y_std > 10.0
        # testovaci cast se standardizuje statistikami z treninku, proto neni presne 0 / 1
        assert abs(x_te.mean()) > 1e-12

    def test_diabetes_je_reprodukovatelne(self) -> None:
        """Stejny seed dava stejne rozdeleni."""
        a = load_diabetes_data(random_state=7)
        b = load_diabetes_data(random_state=7)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[3], b[3])

    def test_sinusovka(self) -> None:
        """Sinusovka: tvar ``(n,)``, serazene ``x`` v ``[-pi, pi]``, sum odpovida ``noise``."""
        x, y = make_sinusoid(n=300, noise=0.1, seed=0)
        assert x.shape == (300,) and y.shape == (300,)
        assert np.all(np.diff(x) >= 0)
        assert x.min() >= -np.pi and x.max() <= np.pi
        assert 0.05 < np.std(y - np.sin(x)) < 0.2
        x2, y2 = make_sinusoid(n=300, noise=0.1, seed=0)
        np.testing.assert_array_equal(y, y2)

    def test_chronological_split_nic_nemicha(self) -> None:
        """Spojeni obou casti dava puvodni radu, trenink je zacatek a test konec."""
        rada = np.arange(100.0)
        train, test = chronological_split(rada, train_frac=0.8)
        assert len(train) == 80 and len(test) == 20
        np.testing.assert_array_equal(np.concatenate([train, test]), rada)
        assert train.max() < test.min()

    def test_random_split_zachova_pary(self) -> None:
        """``random_split`` zachova dvojice ``x`` - ``y`` a nic neztrati ani nezdvoji."""
        x = np.arange(50.0)
        y = 2.0 * x + 1.0
        x_tr, y_tr, x_te, y_te = random_split(x, y, test_size=0.2, random_state=0)
        assert len(x_te) == 10 and len(x_tr) == 40
        np.testing.assert_allclose(y_tr, 2.0 * x_tr + 1.0)
        np.testing.assert_allclose(y_te, 2.0 * x_te + 1.0)
        np.testing.assert_array_equal(np.sort(np.concatenate([x_tr, x_te])), x)

    def test_load_config(self) -> None:
        """``config.yaml`` se nacte do typovanych dataclass a projde validaci."""
        cfg = load_config(CONFIG_YAML)
        validate_config(cfg)
        assert cfg.regression.hidden_dims == [32, 16]
        assert cfg.forecast.window_size >= 1
        assert cfg.regression.training.seed == cfg.seed
        assert cfg.forecast.training.seed == cfg.seed
        assert cfg.regression.training.lr > 0

    @pytest.mark.parametrize("zmena, klic", [
        (lambda c: setattr(c.regression.training, "lr", -0.1), "regression.lr"),
        (lambda c: setattr(c.regression.training, "epochs", 0), "regression.epochs"),
        (lambda c: setattr(c.forecast.training, "batch_size", 0), "forecast.batch_size"),
        (lambda c: setattr(c.regression, "hidden_dims", []), "regression.hidden_dims"),
        (lambda c: setattr(c.forecast, "hidden_dims", [8, 0]), "forecast.hidden_dims"),
        (lambda c: setattr(c.regression, "test_size", 1.5), "test_size"),
        (lambda c: setattr(c.forecast, "window_size", 0), "window_size"),
        (lambda c: setattr(c.forecast, "train_frac", 0.0), "train_frac"),
        (lambda c: setattr(c.paths, "graphs_dir", ""), "paths.graphs_dir"),
        (lambda c: setattr(c.paths, "temperature_csv", ""), "paths.temperature_csv"),
        (lambda c: setattr(c.sinusoid, "n_samples", 3), "sinusoid.n_samples"),
        (lambda c: setattr(c.sinusoid, "noise", -0.1), "sinusoid.noise"),
        (lambda c: setattr(c.diagnostics, "divergence_factor", 0.5),
         "diagnostics.divergence_factor"),
        (lambda c: setattr(c.diagnostics, "overfit_rise", 0.0), "diagnostics.overfit_rise"),
        (lambda c: setattr(c.diagnostics, "overfit_min_fraction", 1.5),
         "diagnostics.overfit_min_fraction"),
        (lambda c: setattr(c.diagnostics, "load_tolerance", 0.0),
         "diagnostics.load_tolerance"),
        (lambda c: setattr(c.diagnostics, "zero_error_tolerance", -1.0),
         "diagnostics.zero_error_tolerance"),
        (lambda c: setattr(c.diagnostics, "normality_alpha", 1.0),
         "diagnostics.normality_alpha"),
    ])
    def test_validate_config_spatne_hodnoty(
        self, zmena: Callable[[ExperimentConfig], None], klic: str,
    ) -> None:
        """Spatna hodnota vyhodi ``ValueError`` s nazvem klice."""
        cfg = copy.deepcopy(load_config(CONFIG_YAML))
        zmena(cfg)
        with pytest.raises(ValueError, match=klic):
            validate_config(cfg)

    def test_load_config_nove_sekce(self) -> None:
        """``load_config`` nacte sekce ``paths``, ``sinusoid`` a ``diagnostics``."""
        cfg = load_config(CONFIG_YAML)
        assert cfg.paths.graphs_dir and cfg.paths.models_dir and cfg.paths.logs_dir
        assert cfg.paths.temperature_csv.endswith(".csv")
        assert cfg.sinusoid.n_samples >= 10
        assert cfg.sinusoid.noise >= 0.0
        diag = cfg.diagnostics
        assert diag.divergence_factor > 1.0
        assert diag.overfit_rise > 0.0
        assert 0.0 < diag.overfit_min_fraction <= 1.0
        assert diag.load_tolerance > 0.0
        assert diag.zero_error_tolerance >= 0.0
        assert 0.0 < diag.normality_alpha < 1.0

    def test_load_config_chybejici_klic(self, tmp_path: Path) -> None:
        """Konfigurace bez povinnych klicu vyhodi ``ValueError``."""
        soubor = tmp_path / "config.yaml"
        soubor.write_text("regression:\n  lr: 0.01\nseed: 1\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_config(str(soubor))


# --------------------------------------------------------------------------- #
#  Pomocny kod pipeline (utils/)                                              #
# --------------------------------------------------------------------------- #
DIAG = DiagnosticsConfig(
    divergence_factor=10.0, overfit_rise=0.10, overfit_min_fraction=0.9,
    load_tolerance=1e-6, zero_error_tolerance=1e-12, normality_alpha=0.05,
)


def _historie(train: list[float], val: list[float]) -> dict[str, list[float]]:
    """Umela historie uceni ve formatu ``Trainer.fit``."""
    return {"epoch": list(range(1, len(train) + 1)), "train_loss": train, "val_loss": val}


def _historie_s_preucenim() -> dict[str, list[float]]:
    """100 epoch: validace klesa do epochy 50 (1.0) a pak roste na 1.5, trenink klesa."""
    train = np.linspace(1.0, 0.2, 100).tolist()
    val = np.concatenate([np.linspace(2.0, 1.0, 50), np.linspace(1.01, 1.5, 50)]).tolist()
    return _historie(train, val)


class TestPipelineUtils:
    """Pomocny kod ``utils/`` (nezavisly na studentskem kodu)."""

    def test_divergence_nekonecna_ztrata(self) -> None:
        """Nekonecna trenovaci ztrata vyhodi ``UceniDivergovalo``."""
        historie = _historie([1.0, 0.5, float("inf")], [1.0, 0.6, float("inf")])
        with pytest.raises(UceniDivergovalo):
            kontrola_uceni(historie, DIAG)

    def test_divergence_nan(self) -> None:
        """``NaN`` ve ztrate vyhodi ``UceniDivergovalo``."""
        with pytest.raises(UceniDivergovalo):
            kontrola_uceni(_historie([1.0, float("nan")], [1.0, float("nan")]), DIAG)

    def test_divergence_dvacetinasobny_narust(self) -> None:
        """Konecna trenovaci ztrata 20x nad ztratou 1. epochy vyhodi ``UceniDivergovalo``."""
        with pytest.raises(UceniDivergovalo):
            kontrola_uceni(_historie([1.0, 5.0, 20.0], [1.0, 5.0, 20.0]), DIAG)

    def test_preuceni_vypise_pozor(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Rostouci validacni ztrata pri klesajici trenovaci vypise ``[POZOR]`` a epochu minima."""
        kontrola_uceni(_historie_s_preucenim(), DIAG)
        out = capsys.readouterr().out
        assert "nejnizsi validacni ztrata 1.0000 v epose 50 z 100" in out
        assert "[POZOR]" in out and "preucuje" in out
        assert "diagnostics.overfit_rise" in out

    def test_klesajici_ztrata_bez_preuceni(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Klesajici trenovaci i validacni ztrata nevypise ``[POZOR]`` ani nevyhodi vyjimku."""
        train = np.linspace(1.0, 0.2, 100).tolist()
        val = np.linspace(1.2, 0.3, 100).tolist()
        kontrola_uceni(_historie(train, val), DIAG)
        out = capsys.readouterr().out
        assert "nejnizsi validacni ztrata" in out
        assert "[POZOR]" not in out

    def test_neklesajici_trenovaci_ztrata(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Neklesajici trenovaci ztrata vypise upozorneni na ``_train_step``, ne na preuceni."""
        kontrola_uceni(_historie([1.0] * 20, [1.0] * 20), DIAG)
        out = capsys.readouterr().out
        assert "Trenovaci ztrata neklesa" in out
        assert "preucuje" not in out

    def test_prah_overfit_rise_se_cte_z_configu(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Zmena ``diagnostics.overfit_rise`` zmeni vysledek (prah se cte z configu)."""
        kontrola_uceni(_historie_s_preucenim(), DIAG)                  # rise 0.10 -> varovani
        assert "[POZOR]" in capsys.readouterr().out
        mirny = dataclasses.replace(DIAG, overfit_rise=1.0)            # 1.5 < 2 * 1.0
        kontrola_uceni(_historie_s_preucenim(), mirny)
        assert "[POZOR]" not in capsys.readouterr().out

    def test_prah_overfit_min_fraction_se_cte_z_configu(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Zmena ``diagnostics.overfit_min_fraction`` zmeni vysledek (minimum v epose 50 z 100)."""
        pozdni = dataclasses.replace(DIAG, overfit_min_fraction=0.4)   # 50 >= 0.4 * 100
        kontrola_uceni(_historie_s_preucenim(), pozdni)
        assert "[POZOR]" not in capsys.readouterr().out

    def test_prah_divergence_factor_se_cte_z_configu(self) -> None:
        """Zmena ``diagnostics.divergence_factor`` zmeni vysledek (20x narust s prahem 50)."""
        historie = _historie([1.0, 5.0, 20.0], [1.0, 5.0, 20.0])
        with pytest.raises(UceniDivergovalo):
            kontrola_uceni(historie, DIAG)
        kontrola_uceni(historie, dataclasses.replace(DIAG, divergence_factor=50.0))

    def test_hlasky_maji_znacky(self, capsys: pytest.CaptureFixture[str]) -> None:
        """``preskoceno`` a ``chyba_dat`` vypisuji pevne znacky pipeline."""
        preskoceno("duvod")
        chyba_dat(FileNotFoundError("chybi soubor"))
        out = capsys.readouterr().out
        assert "[PRESKOCENO] duvod" in out
        assert "[CHYBA DAT] chybi soubor" in out

    def test_predikuj_vraci_1d_pole(self) -> None:
        """``predikuj`` vraci 1D pole NumPy delky datasetu (model ``nn.Linear``)."""
        trainer = Trainer(nn.Linear(2, 1), _cfg(), log_dir=None)
        ds = TabularDataset(np.zeros((7, 2)), np.zeros(7))
        y = predikuj(trainer, ds)
        assert isinstance(y, np.ndarray)
        assert y.shape == (7,)


# --------------------------------------------------------------------------- #
#  End-to-end                                                                 #
# --------------------------------------------------------------------------- #
class TestUceniEndToEnd:
    """Studentova ``MLP`` a ``Trainer`` na jednoduche zavislosti."""

    @STUB
    def test_mlp_se_nauci_jednoduchou_zavislost(self) -> None:
        """Po kratkem uceni ma ``MLP`` na testu ``R^2 > 0.9`` (studentovo ``r2``)."""
        rng = np.random.default_rng(0)
        x = rng.uniform(-2.0, 2.0, size=200)
        y = 0.5 * x ** 2 + x + rng.normal(0.0, 0.05, size=200)
        x_tr, y_tr, x_te, y_te = random_split(x, y, test_size=0.25, random_state=0)

        torch.manual_seed(0)
        model = MLP(1, [16])
        trainer = Trainer(model, _cfg(lr=0.05, epochs=150, batch_size=16), log_dir=None)
        train_ds = TabularDataset(x_tr, y_tr)
        test_ds = TabularDataset(x_te, y_te)
        trainer.fit(DataLoader(train_ds, batch_size=16, shuffle=True),
                    DataLoader(test_ds, batch_size=16, shuffle=False))

        y_pred = trainer.predict(torch.as_tensor(x_te, dtype=torch.float32).reshape(-1, 1))
        assert r2(y_te, y_pred.numpy().reshape(-1)) > 0.9


# --------------------------------------------------------------------------- #
#  Skutecna rada teplot                                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not os.path.exists(TEMPERATURE_PATH),
                    reason=f"chybi soubor s teplotami {TEMPERATURE_CSV}")
class TestForecastNaTeplotach:
    """Predpoved na skutecne rade dennich minimalnich teplot (Melbourne)."""

    WINDOW = 14

    @staticmethod
    def _rozdelena_rada() -> tuple[np.ndarray, np.ndarray]:
        """Nacte radu teplot a rozdeli ji chronologicky 80 / 20."""
        return chronological_split(load_temperature_series(TEMPERATURE_PATH), train_frac=0.8)

    def test_nacteni_rady(self) -> None:
        """Rada ma 3650 hodnot, zadne ``NaN``, prvni hodnota je 20.7."""
        rada = load_temperature_series(TEMPERATURE_PATH)
        assert rada.shape == (3650,)
        assert not np.isnan(rada).any()
        assert rada[0] == pytest.approx(20.7)
        assert rada[-1] == pytest.approx(13.0)

    def test_chronologicke_rozdeleni(self) -> None:
        """Rozdeleni 2920 / 730, bez michani: soucet casti je puvodni rada."""
        rada = load_temperature_series(TEMPERATURE_PATH)
        train, test = chronological_split(rada, train_frac=0.8)
        assert len(train) == 2920 and len(test) == 730
        np.testing.assert_array_equal(np.concatenate([train, test]), rada)

    @STUB
    def test_okna_jsou_zarovnana_s_baseline(self) -> None:
        """Na skutecne rade: ``baseline[i]`` je posledni hodnota okna ``i``, delky souhlasi."""
        _, test = self._rozdelena_rada()
        ds = WindowedDataset(test, self.WINDOW)
        baseline = persistence_baseline(test, self.WINDOW)
        assert len(ds) == len(test) - self.WINDOW
        assert baseline.shape == (len(ds),)
        for i in (0, 1, len(ds) // 2, len(ds) - 1):
            x, y = ds[i]
            assert baseline[i] == pytest.approx(float(x[-1]), abs=1e-4)
            assert float(y[0]) == pytest.approx(test[self.WINDOW + i], abs=1e-4)

    @STUB
    def test_mae_persistence_je_v_rozumnem_rozsahu(self) -> None:
        """MAE persistence na testu odpovida prumernemu dennimu skoku a je mezi 1 a 3 C."""
        _, test = self._rozdelena_rada()
        cile = test[self.WINDOW:]
        baseline = persistence_baseline(test, self.WINDOW)
        chyba = mae(cile, baseline)
        assert chyba == pytest.approx(float(np.mean(np.abs(np.diff(test)[self.WINDOW - 1:]))))
        assert 1.0 < chyba < 3.0
