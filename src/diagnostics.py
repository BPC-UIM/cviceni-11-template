"""Diagnostika rezidui regrese -- vse PREDVYPLNENE.

Rezidua `e_i = y_true_i - y_pred_i` jsou to, co model nevysvetlil. Dobry model
nechava rezidua, ktera vypadaji jako **nahodny sum**: zhruba normalni, se stredem
v nule, se stejnym rozptylem v cele ose predikci a bez zjevneho vzoru. Cokoli
jineho znamena, ze v datech zustala struktura, kterou model nezachytil.

Trida `ResidualAnalyzer` nabizi jeden test a ctyri grafy:

- `residual_normality` - Shapiruv-Wilkuv test normality rezidui,
- `qq_residuals` - Q-Q graf (kvantily rezidui vs. kvantily normalniho rozdeleni),
- `histogram_residuals` - histogram s krivkou normalniho rozdeleni,
- `residuals_vs_prediction` - rezidua proti predikci (nejdulezitejsi graf),
- `residuals_vs_x` - rezidua proti jednomu priznaku.

Vsechny grafy pouzivaji neinteraktivni backend ``Agg``: figuru sestavi,
volitelne ulozi do ``save_path`` (vcetne vytvoreni adresare) a vzdy ji zavrou.
Funkce ``plt.show`` se nikdy nevola.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

import matplotlib

matplotlib.use("Agg")  # neinteraktivni backend, vykreslujeme jen do souboru

import matplotlib.pyplot as plt  # noqa: E402  (musi az po matplotlib.use)
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

# --- Paleta (shodna s dataio/plotting.py) ---------------------------------------
_TRAIN_COLOR = "#2a78d6"     # modra
_TEST_COLOR = "#eb6834"      # oranzova
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_HAIRLINE = "#c3c2b7"


def _save_and_close(fig: plt.Figure, save_path: str | None) -> None:
    """Pomocna funkce: ulozi figuru do ``save_path`` a zavre ji (PREDVYPLNENO).

    Pokud je ``save_path`` ``None``, figura se pouze zavre. Nadrazeny
    adresar se v pripade potreby vytvori.

    Parametry
    ---------
    fig : plt.Figure
        Hotova figura.
    save_path : str | None
        Cesta k vystupnimu .png; ``None`` = neukladat.
    """
    if save_path is not None:
        parent = os.path.dirname(save_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        fig.savefig(save_path, dpi=110, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _style_axes(ax: plt.Axes) -> None:
    """Sjednoti vzhled os: tlumene ramecky a popisky (PREDVYPLNENO).

    Parametry
    ---------
    ax : plt.Axes
        Osy, ktere se upravuji.
    """
    for spine in ax.spines.values():
        spine.set_color(_HAIRLINE)
    ax.tick_params(colors=_INK_SECONDARY, labelsize=9)
    ax.xaxis.label.set_color(_INK_SECONDARY)
    ax.yaxis.label.set_color(_INK_SECONDARY)
    ax.title.set_color(_INK)


@contextmanager
def _figure_scope(save_path: str | None,
                  figsize: tuple[float, float] = (7.0, 4.6)) -> Iterator[plt.Axes]:
    """Kontextovy spravce: vytvori figuru, po uspesnem kresleni ji ulozi (PREDVYPLNENO).

    Figura se zavre VZDY (i kdyz behem kresleni vznikne vyjimka), aby v pameti
    nezustavaly otevrene figury.

    Parametry
    ---------
    save_path : str | None
        Cesta k vystupnimu .png; ``None`` = neukladat.
    figsize : tuple[float, float]
        Rozmery figury v palcich.

    Vraci
    -----
    Iterator[plt.Axes]
        Jedny osy ve spolecnem stylu.
    """
    fig, ax = plt.subplots(figsize=figsize)
    try:
        fig.patch.set_facecolor(_SURFACE)
        ax.set_facecolor(_SURFACE)
        yield ax
        _style_axes(ax)
        _save_and_close(fig, save_path)
    finally:
        plt.close(fig)


class ResidualAnalyzer:
    """Rozbor rezidui modelu: test normality a ctyri diagnosticke grafy (PREDVYPLNENO).

    Atributy
    --------
    y_true : np.ndarray
        Skutecne hodnoty, tvar `(n,)`, float64.
    y_pred : np.ndarray
        Predikce, tvar `(n,)`, float64.
    residuals_ : np.ndarray
        Rezidua `y_true - y_pred`, tvar `(n,)`.
    """

    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Ulozi data (zplosti na 1D float64), zkontroluje je a spocte rezidua (PREDVYPLNENO).

        Parametry
        ---------
        y_true : np.ndarray
            Skutecne hodnoty (libovolny tvar s `n` prvky).
        y_pred : np.ndarray
            Predikce se stejnym poctem prvku.

        Vyjimky
        -------
        ``ValueError``:
            Pokud se delky `y_true` a `y_pred` lisi, pokud je mene nez 3 hodnoty
            nebo pokud rezidua obsahuji NaN / inf (typicky po divergenci uceni).
        """
        self.y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
        self.y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
        if self.y_true.shape != self.y_pred.shape:
            raise ValueError(
                f"y_true a y_pred musi mit stejnou delku, "
                f"dostali jsme {self.y_true.shape[0]} a {self.y_pred.shape[0]}."
            )
        if self.y_true.shape[0] < 3:
            raise ValueError(
                f"Pro diagnostiku rezidui jsou potreba alespon 3 hodnoty, "
                f"dostali jsme {self.y_true.shape[0]}."
            )
        self.residuals_ = self.compute_residuals()
        if not np.all(np.isfinite(self.residuals_)):
            raise ValueError(
                "Rezidua obsahuji NaN/inf (predikce nebo skutecne hodnoty nejsou konecne), "
                "uceni pravdepodobne divergovalo, zkuste mensi lr."
            )

    def compute_residuals(self) -> np.ndarray:
        """Vypocte rezidua `y_true - y_pred` (PREDVYPLNENO).

        Kladne reziduum = model podhodnotil, zaporne = nadhodnotil.

        Navratova hodnota
        -----------------
        np.ndarray
            Rezidua, tvar `(n,)`.
        """
        return self.y_true - self.y_pred

    def residual_normality(self, alpha: float = 0.05) -> dict[str, float | bool]:
        """Shapiruv-Wilkuv test normality rezidui (PREDVYPLNENO).

        Nulova hypoteza: rezidua pochazeji z normalniho rozdeleni. Je-li
        `p_value > alpha`, nulovou hypotezu nezamitame (`normal = True`).

        Omezeni: pro `n > 5000` je p-hodnota Shapirova-Wilkova testu jen
        priblizna. Na velkych souborech navic test zamita i prakticky
        bezvyznamne odchylky od normality, proto ho vzdy cteme spolecne
        s Q-Q grafem a histogramem.

        Parametry
        ---------
        alpha : float
            Hladina vyznamnosti.

        Navratova hodnota
        -----------------
        dict[str, float | bool]
            Klice `"statistic"` (float), `"p_value"` (float), `"normal"` (bool).

        Vyjimky
        -------
        ``ValueError``:
            Pokud jsou vsechna rezidua shodna (test normality nema smysl).
        """
        if np.ptp(self.residuals_) == 0.0:
            raise ValueError(
                "Vsechna rezidua jsou shodna (nulovy rozptyl), test normality nelze spocitat."
            )
        statistic, p_value = stats.shapiro(self.residuals_)
        return {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "normal": bool(p_value > alpha),
        }

    def qq_residuals(self, save_path: str | None = None) -> None:
        """Q-Q graf rezidui proti normalnimu rozdeleni (PREDVYPLNENO).

        Jak cist: body (usporadana rezidua vs. teoreticke kvantily normalniho
        rozdeleni) maji lezet na primce.

        - V poradku: body kopiruji primku v cele delce, rezidua jsou
          priblizne normalni.
        - Varovny signal, oblouk ("banan"): oba konce lezi na STEJNE strane
          primky = sikmost (oba konce nad primkou = sesikmeni doprava,
          oba pod primkou = doleva).
        - Varovny signal, tvar S: konce lezi na OPACNYCH stranach primky =
          chvosty. Pravy konec nad primkou a levy pod ni = tezke chvosty
          (vic extremnich chyb nez u normalniho rozdeleni); obracene = lehke
          chvosty.

        Parametry
        ---------
        save_path : str | None
            Cesta k vystupnimu .png; ``None`` = neukladat.
        """
        with _figure_scope(save_path, (5.6, 5.0)) as ax:
            (theoretical, ordered), (slope, intercept, _) = stats.probplot(
                self.residuals_, dist="norm"
            )
            ax.scatter(theoretical, ordered, s=18, color=_TRAIN_COLOR, alpha=0.8,
                       edgecolors="none", label="rezidua")
            line_x = np.array([theoretical.min(), theoretical.max()])
            ax.plot(line_x, slope * line_x + intercept, color=_TEST_COLOR, lw=2.0, ls="--",
                    label="prolozena primka (normalni rozdeleni)")
            ax.set_xlabel("teoreticke kvantily (normalni rozdeleni)")
            ax.set_ylabel("usporadana rezidua")
            ax.set_title("Q-Q graf rezidui")
            ax.grid(True, color=_HAIRLINE, lw=0.5, alpha=0.6)
            ax.legend(frameon=False, fontsize=9, loc="upper left")

    def histogram_residuals(self, save_path: str | None = None, bins: int = 20) -> None:
        """Histogram rezidui s krivkou normalniho rozdeleni (PREDVYPLNENO).

        Krivka ma stejny prumer a smerodatnou odchylku jako rezidua.

        Jak cist: histogram ma zhruba kopirovat zvonovou krivku se stredem
        poblize nuly.

        - V poradku: symetricky zvon, stred v nule.
        - Varovny signal: stred posunuty od nuly = systematicka chyba (bias);
          asymetrie nebo dlouhy chvost na jedne strane; dva vrcholy =
          model mezi sebou micha dve ruzne skupiny dat.

        Parametry
        ---------
        save_path : str | None
            Cesta k vystupnimu .png; ``None`` = neukladat.
        bins : int
            Pocet intervalu histogramu.
        """
        with _figure_scope(save_path) as ax:
            mean = float(np.mean(self.residuals_))
            std = float(np.std(self.residuals_))
            ax.hist(self.residuals_, bins=bins, density=True, color=_TRAIN_COLOR, alpha=0.75,
                    edgecolor=_SURFACE, label="rezidua")
            if std > 0:
                grid = np.linspace(self.residuals_.min(), self.residuals_.max(), 200)
                ax.plot(grid, stats.norm.pdf(grid, loc=mean, scale=std), color=_TEST_COLOR,
                        lw=2.0, label="normalni rozdeleni (stejny prumer a odchylka)")
            ax.axvline(0.0, color=_INK_SECONDARY, lw=1.0, ls=":")
            ax.set_xlabel("reziduum (y_true - y_pred)")
            ax.set_ylabel("hustota")
            ax.set_title("Histogram rezidui")
            ax.grid(True, color=_HAIRLINE, lw=0.5, alpha=0.6)
            ax.set_ylim(top=ax.get_ylim()[1] * 1.3)  # misto pro legendu nad sloupci
            ax.legend(frameon=False, fontsize=9, loc="upper right")

    def residuals_vs_prediction(self, save_path: str | None = None) -> None:
        """Rezidua proti predikci s nulou a pasem +-2 sigma (PREDVYPLNENO).

        Jak cist: body maji tvorit rovnomerny "mrak" kolem nuly, zhruba 95 %
        jich lezi uvnitr pasu +-2 sigma.

        - V poradku: mrak bez tvaru, stejne siroky vlevo i vpravo.
        - Varovny signal: **trychtyr** (rozptyl roste nebo klesa s predikci) =
          heteroskedasticita; **oblouk** nebo vlna = model nezachytil
          nelinearitu; osamocene body daleko za pasem = odlehle hodnoty.

        Parametry
        ---------
        save_path : str | None
            Cesta k vystupnimu .png; ``None`` = neukladat.
        """
        with _figure_scope(save_path) as ax:
            self._scatter_residuals(ax, self.y_pred)
            ax.set_xlabel("predikce")
            ax.set_ylabel("reziduum (y_true - y_pred)")
            ax.set_title("Rezidua vs. predikce")

    def residuals_vs_x(self, x: np.ndarray, save_path: str | None = None,
                       x_label: str = "x") -> None:
        """Rezidua proti jednomu priznaku `x` s nulou a pasem +-2 sigma (PREDVYPLNENO).

        Jak cist: stejne jako u grafu rezidui proti predikci, ale osa x je
        jeden vybrany priznak. Pomaha najit, ktery priznak model zpracovava
        spatne.

        - V poradku: mrak bez tvaru kolem nuly.
        - Varovny signal: oblouk nebo vlna = vztah mezi priznakem a cilem
          neni linearni a model ho nezachytil; zmena sirky mraku =
          heteroskedasticita zavisla na priznaku.

        Parametry
        ---------
        x : np.ndarray
            Hodnoty priznaku, `n` prvku (stejny pocet jako rezidui).
        save_path : str | None
            Cesta k vystupnimu .png; ``None`` = neukladat.
        x_label : str
            Popisek osy x (jmeno priznaku).

        Vyjimky
        -------
        ``ValueError``:
            Pokud ma `x` jiny pocet prvku nez rezidua.
        """
        x_values = np.asarray(x, dtype=np.float64).reshape(-1)
        if x_values.shape != self.residuals_.shape:
            raise ValueError(
                f"x musi mit {self.residuals_.shape[0]} prvku, "
                f"dostali jsme {x_values.shape[0]}."
            )
        with _figure_scope(save_path) as ax:
            self._scatter_residuals(ax, x_values)
            ax.set_xlabel(x_label)
            ax.set_ylabel("reziduum (y_true - y_pred)")
            ax.set_title(f"Rezidua vs. {x_label}")

    def _scatter_residuals(self, ax: plt.Axes, horizontal: np.ndarray) -> None:
        """Pomocna metoda: body rezidui, nula, pas +-2 sigma a legenda pod grafem (PREDVYPLNENO).

        Parametry
        ---------
        ax : plt.Axes
            Osy, do kterych se kresli.
        horizontal : np.ndarray
            Hodnoty na ose x, tvar `(n,)`.
        """
        sigma = float(np.std(self.residuals_))
        ax.scatter(horizontal, self.residuals_, s=18, color=_TRAIN_COLOR, alpha=0.7,
                   edgecolors="none", label="rezidua")
        ax.axhline(0.0, color=_INK_SECONDARY, lw=1.2, label="nula")
        ax.axhspan(-2.0 * sigma, 2.0 * sigma, color=_TEST_COLOR, alpha=0.12,
                   label="pas +-2 sigma")
        ax.grid(True, color=_HAIRLINE, lw=0.5, alpha=0.6)
        # legenda mimo osy (pod grafem), aby neprekryvala body
        ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.16),
                  ncol=3)
