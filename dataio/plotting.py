"""Vykreslovani vysledku cviceni 11 -- vse PREDVYPLNENE.

Pet grafu z pipeline:

* ``plot_regression_fit`` -- zasumena data a krivka naucene site (1D regrese).
* ``plot_predicted_vs_actual`` -- predikce proti skutecnosti (vice priznaku).
* ``plot_loss_curve`` -- prubeh trenovaci a validacni ztraty (krivka uceni).
* ``plot_forecast`` -- predpoved a skutecnost casove rady (cely test + detail).
* ``plot_residuals_in_time`` -- rezidua predpovedi v case.

Vsechny funkce pouzivaji neinteraktivni backend ``Agg``: figuru sestavi,
volitelne ulozi do ``save_path`` (vcetne vytvoreni nadrazeneho adresare) a
vzdy ji zavrou, i kdyz behem kresleni nebo ukladani nastane chyba. Funkce
``plt.show`` se nikdy nevola.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

import matplotlib

matplotlib.use("Agg")  # neinteraktivni backend, vykreslujeme jen do souboru

import matplotlib.pyplot as plt  # noqa: E402  (musi az po matplotlib.use)
import numpy as np  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

# --- Paleta (shodna s Cvicenimi 09 a 10) --------------------------------------
_TRAIN_COLOR = "#2a78d6"     # modra
_TEST_COLOR = "#eb6834"      # oranzova
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_MUTED = "#898781"
_HAIRLINE = "#c3c2b7"

_DETAIL_DAYS = 120           # delka detailu v grafu predpovedi
_LOG_MIN_RATIO = 10.0        # log. osa ztraty jen pri rozsahu aspon o rad


@contextmanager
def _figure_scope(fig: plt.Figure) -> Iterator[plt.Figure]:
    """Pomocny kontextovy spravce: figuru po skonceni bloku VZDY zavre.

    Zavre ji i pri vyjimce (spatna data, neplatna pripona), takze se
    neodkladaji otevrene figury v ``plt.get_fignums()``.
    """
    try:
        yield fig
    finally:
        plt.close(fig)


def _save_and_close(fig: plt.Figure, save_path: str | None) -> None:
    """Pomocna funkce: ulozi figuru do ``save_path`` a zavre ji.

    Pokud je ``save_path`` ``None``, figura se pouze zavre. Nadrazeny
    adresar se v pripade potreby vytvori. Figura se zavre i pri chybe ukladani.
    """
    try:
        if save_path is not None:
            parent = os.path.dirname(save_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            fig.savefig(save_path, dpi=110, bbox_inches="tight", facecolor=fig.get_facecolor())
    finally:
        plt.close(fig)


def _style_axes(ax: plt.Axes) -> None:
    """Sjednoti vzhled os: tlumene ramecky, mrizka a popisky."""
    ax.set_facecolor(_SURFACE)
    for spine in ax.spines.values():
        spine.set_color(_HAIRLINE)
    ax.tick_params(colors=_INK_SECONDARY, labelsize=9)
    ax.xaxis.label.set_color(_INK_SECONDARY)
    ax.yaxis.label.set_color(_INK_SECONDARY)
    ax.title.set_color(_INK)
    ax.grid(True, color=_HAIRLINE, lw=0.5, alpha=0.6)


def _add_headroom(ax: plt.Axes, fraction: float = 0.18) -> None:
    """Pomocna funkce: zvetsi horni mez osy y, aby legenda neprekryvala data."""
    low, high = ax.get_ylim()
    ax.set_ylim(low, high + fraction * (high - low))


def plot_regression_fit(
    x: np.ndarray,
    y: np.ndarray,
    x_grid: np.ndarray,
    y_pred: np.ndarray,
    save_path: str | None = None,
    ) -> None:
    """Vykresli zasumena data a krivku naucene site.

    Parametry
    ---------
    x, y:
        Data (1D pole), vykresli se jako body.
    x_grid:
        Husta mrizka hodnot ``x`` (serazena), na ktere se vyhodnotila sit.
    y_pred:
        Predikce site na ``x_grid`` (1D pole).
    save_path:
        Cesta k vystupnimu .png; ``None`` = neukladat.
    """
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    with _figure_scope(fig):
        fig.patch.set_facecolor(_SURFACE)
        ax.scatter(x, y, s=14, color=_MUTED, alpha=0.7, label="data")
        ax.plot(x_grid, y_pred, color=_TEST_COLOR, lw=2.2, label="sit (MLP)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Prolozeni dat neuronovou siti")
        _style_axes(ax)
        ax.legend(frameon=False, fontsize=9, loc="best")
        _save_and_close(fig, save_path)


def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str | None = None,
    ) -> None:
    """Vykresli predikci proti skutecnosti s diagonalou ``y = x``.

    Body na diagonale jsou presne predpovedi; vzdalenost od ni je chyba.
    Systematicky sklon nebo oblouk kolem diagonaly znamena zkreslenou predikci.

    Parametry
    ---------
    y_true:
        Skutecne hodnoty (1D pole).
    y_pred:
        Predikovane hodnoty (1D pole).
    save_path:
        Cesta k vystupnimu .png; ``None`` = neukladat.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    with _figure_scope(fig):
        lo = float(min(y_true.min(), y_pred.min()))
        hi = float(max(y_true.max(), y_pred.max()))
        pad = 0.05 * (hi - lo if hi > lo else 1.0)

        fig.patch.set_facecolor(_SURFACE)
        ax.scatter(y_true, y_pred, s=18, color=_TRAIN_COLOR, alpha=0.65, label="vzorky")
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=_INK_SECONDARY, lw=1.2,
                ls="--", label="ideal (y = x)")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal")
        ax.set_xlabel("skutecna hodnota")
        ax.set_ylabel("predikovana hodnota")
        ax.set_title("Predikce proti skutecnosti (testovaci data)")
        _style_axes(ax)
        ax.legend(frameon=False, fontsize=9, loc="upper left")
        _save_and_close(fig, save_path)


def plot_loss_curve(
    history: dict[str, list[float]],
    save_path: str | None = None,
    title: str = "Krivka uceni",
    ) -> None:
    """Vykresli prubeh trenovaci a validacni ztraty po epochach.

    Osa y je logaritmicka jen tehdy, kdyz jsou vsechny hodnoty kladne a konecne
    a nejvetsi je aspon 10x vetsi nez nejmensi (ztrata se meni pres rad);
    jinak je linearni, aby mela citelne popisky. Rostouci validacni krivka
    pri klesajici trenovaci je znamka preuceni. Legenda je pod grafem, aby
    neprekryvala krivky. Osa epoch ma jen cela cisla; pri jedine epose se
    vykresli body.

    Parametry
    ---------
    history:
        Slovnik z ``Trainer.fit`` s klici ``epoch``, ``train_loss``, ``val_loss``.
    save_path:
        Cesta k vystupnimu .png; ``None`` = neukladat.
    title:
        Nadpis grafu.
    """
    epochs = np.asarray(history["epoch"])
    train_loss = np.asarray(history["train_loss"], dtype=np.float64)
    val_loss = np.asarray(history["val_loss"], dtype=np.float64)
    marker = "o" if len(epochs) == 1 else None

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    with _figure_scope(fig):
        fig.patch.set_facecolor(_SURFACE)
        ax.plot(epochs, train_loss, color=_TRAIN_COLOR, lw=2.0, marker=marker, ms=7,
                label="trenovaci data")
        ax.plot(epochs, val_loss, color=_TEST_COLOR, lw=2.0, ls="--", marker=marker, ms=7,
                label="validacni data")

        all_loss = np.concatenate([train_loss, val_loss])
        if np.all(np.isfinite(all_loss)) and all_loss.min() > 0 \
                and all_loss.max() / all_loss.min() >= _LOG_MIN_RATIO:
            ax.set_yscale("log")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("epocha")
        ax.set_ylabel("ztrata (MSE)")
        ax.set_title(title)
        _style_axes(ax)
        ax.grid(True, axis="y", which="major", color=_HAIRLINE, lw=0.5, alpha=0.6)
        ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.16),
                  ncol=2)
        _save_and_close(fig, save_path)


def plot_forecast(
    y_true_t: np.ndarray,
    y_pred_t: np.ndarray,
    save_path: str | None = None,
    ) -> None:
    """Vykresli skutecnost a predpoved v case ve dvou panelech.

    Horni panel ukazuje cely testovaci usek, dolni detail poslednich
    ``120`` dni, kde je videt, jak predpoved kopiruje skutecny prubeh
    (a o kolik se opozduje za prudkymi zmenami).

    Parametry
    ---------
    y_true_t:
        Skutecne hodnoty rady v case (1D pole).
    y_pred_t:
        Predpovedi na stejne dny (1D pole stejne delky).
    save_path:
        Cesta k vystupnimu .png; ``None`` = neukladat.
    """
    y_true_t = np.asarray(y_true_t).reshape(-1)
    y_pred_t = np.asarray(y_pred_t).reshape(-1)
    fig, (ax_all, ax_detail) = plt.subplots(2, 1, figsize=(9.0, 7.0))
    with _figure_scope(fig):
        if len(y_true_t) != len(y_pred_t):
            raise ValueError(
                f"y_true_t a y_pred_t musi mit stejnou delku, zadano: "
                f"{len(y_true_t)} a {len(y_pred_t)}"
            )
        days = np.arange(len(y_true_t))
        start = max(0, len(days) - _DETAIL_DAYS)
        fig.patch.set_facecolor(_SURFACE)

        ax_all.plot(days, y_true_t, color=_TRAIN_COLOR, lw=1.2, label="skutecnost")
        ax_all.plot(days, y_pred_t, color=_TEST_COLOR, lw=1.2, alpha=0.9, label="predpoved")
        ax_all.axvspan(start, len(days) - 1, color=_HAIRLINE, alpha=0.35, lw=0)
        ax_all.set_xlabel("den (index v testovacim useku)")
        ax_all.set_ylabel("teplota (C)")
        ax_all.set_title("Predpoved teploty -- cely testovaci usek")
        _style_axes(ax_all)
        _add_headroom(ax_all)
        ax_all.legend(frameon=False, fontsize=9, loc="upper right", ncol=2)

        ax_detail.plot(days[start:], y_true_t[start:], color=_TRAIN_COLOR, lw=1.8,
                       marker="o", ms=3, label="skutecnost")
        ax_detail.plot(days[start:], y_pred_t[start:], color=_TEST_COLOR, lw=1.8, ls="--",
                       marker="o", ms=3, label="predpoved")
        ax_detail.set_xlabel("den (index v testovacim useku)")
        ax_detail.set_ylabel("teplota (C)")
        ax_detail.set_title(f"Detail poslednich {len(days) - start} dni (sedy pas nahore)")
        _style_axes(ax_detail)
        _add_headroom(ax_detail)
        ax_detail.legend(frameon=False, fontsize=9, loc="upper right", ncol=2)

        fig.tight_layout(h_pad=2.0)
        _save_and_close(fig, save_path)


def plot_residuals_in_time(
    residuals: np.ndarray,
    save_path: str | None = None,
    ) -> None:
    """Vykresli rezidua predpovedi proti indexu dne s nulou a pasem +-2 sigma.

    Pas ``+-2 sigma`` je vystredeny na NULU (stejne jako v ``ResidualAnalyzer``),
    takze je videt pripadny bias: pokud rezidua lezi vetsinou nad nebo pod
    nulou, model je systematicky posunuty. Prumer rezidui je vyznacen tenkou
    carou a uveden v legende. Dobra predpoved ma rezidua kolem nuly bez
    trendu a bez opakujicich se vzoru; dlouhe useky nad nebo pod nulou
    znamenaji, ze si model neco systematicky nevsiml (napr. sezonu).

    Parametry
    ---------
    residuals:
        Rezidua ``y_true - y_pred`` v case (1D pole).
    save_path:
        Cesta k vystupnimu .png; ``None`` = neukladat.
    """
    residuals = np.asarray(residuals, dtype=np.float64).reshape(-1)
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    with _figure_scope(fig):
        days = np.arange(len(residuals))
        mean = float(residuals.mean())
        sigma = float(residuals.std())

        fig.patch.set_facecolor(_SURFACE)
        ax.axhspan(-2 * sigma, 2 * sigma, color=_TRAIN_COLOR, alpha=0.12, lw=0,
                   label="pas +-2 sigma kolem nuly")
        ax.plot(days, residuals, color=_TRAIN_COLOR, lw=0.9, label="rezidua")
        ax.axhline(0.0, color=_INK_SECONDARY, lw=1.2, ls="--", label="nula")
        ax.axhline(mean, color=_TEST_COLOR, lw=0.9, label=f"prumer rezidui = {mean:.2f}")
        ax.set_xlabel("den (index v testovacim useku)")
        ax.set_ylabel("reziduum (skutecnost - predpoved)")
        ax.set_title("Rezidua predpovedi v case")
        _style_axes(ax)
        ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.16),
                  ncol=4)
        _save_and_close(fig, save_path)
