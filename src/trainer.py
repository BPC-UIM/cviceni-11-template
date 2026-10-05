"""Trenovaci wrapper pro PyTorch: epochy, davky, vyhodnoceni, logovani, ukladani.

Delba zodpovednosti (stejna jako v cv10):

- **Model** (`nn.Module`) umi dopredny pruchod.
- **Trainer** sklada kroky do *procesu*: epochy, davky, sledovani ztraty a logovani.

Rozdil proti cv10: rucni `backward` a `update` jsou nahrazeny volanimi
`loss.backward()` a `optimizer.step()` (viz `Trainer._train_step`, jadro cviceni).

Logovani: pojmenovany logger s `propagate = False` a dvema handlery - konzole
(uroven INFO, milniky uceni) a rotujici soubor `logs/trainer.log` (uroven DEBUG,
zaznam kazde epochy). Do logu se zapisuji jen skalary, nikdy tenzory.
"""

from __future__ import annotations

import logging
import math
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from dataio.config_manager import TrainingConfig

LOG_FILE_NAME = "trainer.log"
_LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)-7s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class _ConsoleHandler(logging.StreamHandler):
    """Konzolovy handler, ktery vzdy pise do aktualniho `sys.stdout` (PREDVYPLNENO).

    Beznemu `StreamHandler` se proud preda jednou pri vytvoreni; pokud jej
    nekdo mezitim vymeni (napr. pytest pri zachytavani vystupu), handler by
    psal do uzavreneho proudu. Tato varianta si proud zjisti pri kazdem zapisu.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Nastavi aktualni `sys.stdout` a zaznam zapise (PREDVYPLNENO)."""
        self.stream = sys.stdout
        super().emit(record)


def get_logger(log_dir: str | None = "logs") -> logging.Logger:
    """Vrati logger modulu, pri prvnim volani ho nakonfiguruje (PREDVYPLNENO).

    Konfigurace probehne **jen jednou** za beh programu (kontrola, zda uz
    logger ma handlery) - opakovane volani tak nezpusobi zdvojeny vystup.

    Parametry
    ---------
    log_dir : str | None, optional
        Adresar pro rotujici soubor logu; `None` znamena jen konzoli.

    Navratova hodnota
    -----------------
    logging.Logger
        Logger tohoto modulu s urovni DEBUG a `propagate = False`.
    """
    logger = logging.getLogger(__name__)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # zadne zdvojeni pres korenovy logger

    console = _ConsoleHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("  [%(levelname)s] %(message)s"))
    logger.addHandler(console)

    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = RotatingFileHandler(
            os.path.join(log_dir, LOG_FILE_NAME),
            maxBytes=1_000_000, backupCount=3, encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(file_handler)

    return logger


class Trainer:
    """Ridi uceni PyTorch modelu gradientnim sestupem (MIX: _train_step, save, load jsou UKOL).

    Atributy
    --------
    model : nn.Module
        Uceny model.
    config : TrainingConfig
        Hyperparametry uceni (`lr`, `epochs`, `batch_size`, `seed`).
    optimizer : torch.optim.Optimizer
        Stochasticky gradientni sestup (`SGD`) nad parametry modelu.
    loss_fn : nn.Module
        Stredni kvadraticka chyba (`nn.MSELoss`).
    logger : logging.Logger
        Logger trenera.
    history_ : dict[str, list[float]] | None
        Prubeh uceni posledniho `fit`; pred prvnim `fit` je `None`.
    """

    def __init__(
        self,
        model: nn.Module,
        config: "TrainingConfig",
        log_dir: str | None = "logs",
    ) -> None:
        """Ulozi model a konfiguraci, vytvori optimalizator, ztratu a logger (PREDVYPLNENO).

        Parametry
        ---------
        model : nn.Module
            Model k uceni.
        config : TrainingConfig
            Hyperparametry uceni.
        log_dir : str | None, optional
            Adresar pro soubor logu; `None` = jen konzole.
        """
        torch.manual_seed(config.seed)
        self.model = model
        self.config = config
        self.optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)
        self.loss_fn = nn.MSELoss()
        self.logger = get_logger(log_dir)
        self.history_: dict[str, list[float]] | None = None

    def _train_step(self, x_batch: torch.Tensor, y_batch: torch.Tensor) -> float:
        """Provede jeden krok uceni na jedne davce (UKOL, jadro cviceni).

        Pet kroku uceni a vraceni ztraty nahrazuje cele rucni zpetne sireni z cv10.
        V tomto poradi:

        1. Vynulujte gradienty optimizeru (v cv10 nemelo obdobu, viz nize).
        2. Provedte dopredny pruchod modelu na `x_batch` (v cv10 `model.forward`).
        3. Spoctete ztratu mezi predikci a `y_batch` pomoci `self.loss_fn`
           (v cv10 `loss.forward`). Tvary predikce a `y_batch` musi byt shodne:
           `nn.MSELoss` pri `(B, 1)` vs `(B,)` s pouhym varovanim (UserWarning)
           broadcastuje na `(B, B)` a pocita nesmysl.
        4. Zavolejte zpetny pruchod ztraty; autograd tak spocita gradienty vsech
           parametru (v cv10 `loss.gradient` + `model.backward`).
        5. Provedte krok optimizeru, ktery posune vahy proti gradientu
           (v cv10 `model.update`).
        6. Vratte hodnotu ztraty davky jako python `float` pres `.item()`.

        Proc krok 1: v PyTorch se gradienty pri kazdem zpetnem pruchodu *pricitaji*
        do `p.grad`, takze bez vynulovani by se hromadily pres davky. V cv10
        `backward` gradienty vracel cerstve, takze nulovani nebylo treba. Poznamka:
        po vynulovani je `p.grad` v aktualnim PyTorchi `None` (ne nulovy tenzor).

        Parametry
        ---------
        x_batch : torch.Tensor
            Vstupy davky, tvar `(B, n_features)`.
        y_batch : torch.Tensor
            Cile davky, tvar `(B, 1)`.

        Navratova hodnota
        -----------------
        float
            Hodnota ztraty na davce **pred** aktualizaci vah.

        Vyjimky
        -------
        ``AssertionError``:
            Pokud se tvar predikce lisi od tvaru cilu.
        """
        # assert  Ověřte, že tvar predikce modelu je shodny s tvarem y_batch
        raise NotImplementedError(
            "Úkol: doplnte jeden trenovaci krok podle ocislovanych kroku v docstringu: "
            "vynulovani gradientu optimizeru, dopredny pruchod modelu, vypocet ztraty, "
            "zpetny pruchod ztraty, krok optimizeru a nakonec vraceni ztraty davky "
            "jako python float."
        )

    def _evaluate(self, loader: "DataLoader") -> float:
        """Vrati prumernou ztratu modelu na celem loaderu, vahy se nemeni (PREDVYPLNENO).

        Prumer je vazeny velikosti davky, aby mensi posledni davka nezkreslila vysledek.

        Parametry
        ---------
        loader : DataLoader
            Davky `(x_batch, y_batch)`.

        Navratova hodnota
        -----------------
        float
            Prumerna ztrata na vsech vzorcich loaderu.

        Vyjimky
        -------
        ``ValueError``:
            Pokud je loader prazdny nebo se tvar predikce lisi od tvaru cilu.
        """
        self.model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for x_batch, y_batch in loader:
                y_pred = self.model(x_batch)
                if y_pred.shape != y_batch.shape:
                    raise ValueError(
                        f"tvar predikce {tuple(y_pred.shape)} se lisi od tvaru cilu "
                        f"{tuple(y_batch.shape)}; ocekavany tvar cilu je (B, 1)"
                    )
                n = x_batch.shape[0]
                total += self.loss_fn(y_pred, y_batch).item() * n
                count += n
        if count == 0:
            raise ValueError("DataLoader je prazdny (0 vzorku), nelze spocitat ztratu")
        return total / count

    def fit(
        self, train_loader: "DataLoader", val_loader: "DataLoader"
    ) -> dict[str, list[float]]:
        """Provede cele uceni a vrati jeho prubeh (PREDVYPLNENO).

        Prubeh jedne epochy: `model.train()`, `_train_step` pro kazdou davku
        (trenovaci ztrata epochy = prumer ztrat davek vazeny velikosti davky),
        pak `_evaluate` na validacnich datech. Neni-li ztrata konecna, uceni
        skonci s varovanim.

        Parametry
        ---------
        train_loader : DataLoader
            Trenovaci davky.
        val_loader : DataLoader
            Validacni (testovaci) davky; uceni je nepouziva.

        Navratova hodnota
        -----------------
        dict[str, list[float]]
            Klice `epoch`, `train_loss`, `val_loss` - jedna hodnota za epochu.

        Vyjimky
        -------
        ``ValueError``:
            Pokud je trenovaci nebo validacni loader prazdny.
        ``TypeError``:
            Pokud `_train_step` nevrati python `float` (napr. vrati tenzor).
        """
        cfg = self.config
        log_every = max(1, cfg.epochs // 10)
        history: dict[str, list[float]] = {"epoch": [], "train_loss": [], "val_loss": []}

        n_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info("Start uceni: %d parametru modelu", n_params)
        self.logger.info(
            "Hyperparametry: lr=%g, epochs=%d, batch_size=%d, seed=%d",
            cfg.lr, cfg.epochs, cfg.batch_size, cfg.seed,
        )

        for epoch in range(1, cfg.epochs + 1):
            self.model.train()
            total, count = 0.0, 0
            for x_batch, y_batch in train_loader:
                n = x_batch.shape[0]
                step_loss = self._train_step(x_batch, y_batch)
                if not isinstance(step_loss, float):
                    raise TypeError(
                        f"_train_step vratil {type(step_loss).__name__}, ocekavan python "
                        "float; vratte loss.item(), ne tenzor"
                    )
                total += step_loss * n
                count += n
            if count == 0:
                raise ValueError("Trenovaci DataLoader je prazdny (0 vzorku)")
            train_loss = total / count
            val_loss = self._evaluate(val_loader)

            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            self.logger.debug(
                "epocha %4d | train %.5f | val %.5f", epoch, train_loss, val_loss
            )
            if epoch == 1 or epoch % log_every == 0:
                self.logger.info(
                    "epocha %4d/%d  train %.4f  val %.4f",
                    epoch, cfg.epochs, train_loss, val_loss,
                )

            if not math.isfinite(train_loss):
                self.logger.warning(
                    "Ztrata neni konecna (%s) - uceni diverguje, koncim. "
                    "Zkuste mensi lr.", train_loss,
                )
                break

        self.logger.info(
            "Konec uceni po %d epochach: train %.4f, val %.4f.",
            history["epoch"][-1], history["train_loss"][-1], history["val_loss"][-1],
        )
        self.model.eval()
        self.history_ = history
        return history

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Vrati predikce modelu bez sledovani gradientu (PREDVYPLNENO).

        Parametry
        ---------
        x : torch.Tensor
            Vstupy, tvar `(n, n_features)`.

        Navratova hodnota
        -----------------
        torch.Tensor
            Predikce, tvar `(n, output_dim)`.
        """
        self.model.eval()
        with torch.no_grad():
            return self.model(x)

    def save(self, path: str) -> None:
        """Ulozi vahy modelu do souboru (UKOL).

        Uklada se jen `state_dict` (slovnik jmeno -> tenzor). `torch.save(model)` by
        zapicklil cely objekt vcetne odkazu na tridu a modul: soubor by pak zavisel na
        zdrojovem kodu a jeho nacteni by mohlo spustit libovolny kod. `state_dict` se
        naopak nacita bezpecne s `weights_only=True`.

        Parametry
        ---------
        path : str
            Cesta k souboru (napr. `models/mlp_diabetes.pt`).
        """
        # assert  Ověřte, že path konci priponou .pt
        raise NotImplementedError(
            "Úkol: ulozte do souboru path slovnik vah modelu (state_dict) pomoci torch.save."
        )

    @classmethod
    def load(cls, path: str, model: nn.Module, config: "TrainingConfig") -> "Trainer":
        """Nacte vahy do nove instance modelu a vrati pro ni `Trainer` (UKOL).

        Parametry
        ---------
        path : str
            Cesta k souboru vytvorenemu pomoci `save`.
        model : nn.Module
            Nova instance modelu STEJNE architektury, jako byla ulozena.
        config : TrainingConfig
            Hyperparametry uceni.

        Navratova hodnota
        -----------------
        Trainer
            Trainer s nactenym modelem.
        """
        # assert  Ověřte, že soubor path existuje
        raise NotImplementedError(
            "Úkol: nactete vahy ze souboru path (torch.load s weights_only=True) do "
            "predaneho modelu pomoci load_state_dict a vratte novy Trainer(model, config)."
        )
