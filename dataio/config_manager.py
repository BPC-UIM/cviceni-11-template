"""Typovana sprava konfigurace nad ``config.yaml`` pro cviceni 11.

Zasada: **vse nastavitelne je v ``config.yaml``** -- cesty, parametry dat,
hyperparametry uceni i prahy diagnostickych kontrol; v kodu pipeline nejsou
zadne "magicke" konstanty.

Modul definuje dataclassy odpovidajici sekcim ``config.yaml`` (``paths``,
``sinusoid``, ``regression``, ``forecast``, ``diagnostics`` a korenovy ``seed``)
a dve funkce: ``load_config`` (naparsuje
YAML, sestavi dataclassy, zvaliduje a vrati) a ``validate_config`` (rozsahove
kontroly s ceskymi chybovymi hlaskami). Cely modul je predvyplneny.

YAML je v ramci sekce plochy (``lr``, ``epochs``, ``batch_size`` primo v sekci);
hyperparametry uceni a korenovy ``seed`` se pri nacteni slozi do
``TrainingConfig`` dane sekce, ktery se predava tride ``Trainer``.

K hodnotam se pristupuje pres atributy (napr. ``cfg.regression.training.lr``),
nikdy ne pres klice slovniku -- preklep v atributu odhali editor/typovy
kontroler staticky, zatimco ``cfg["regression"]["lr"]`` spadne az za behu.

Kazda chyba konfigurace (spatny YAML, chybejici klic, spatny typ ci rozsah)
konci jako ``ValueError`` s ceskou hlaskou; zadna hodnota se tise nepretypovava.

Poznamka: site jsou zamerne male (``hidden_dims: [32, 16]``). Stara verze
cviceni prokladala 1D sinusovku siti 6x70 neuronu, coz je zbytecne hluboke
a spatne se uci.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import yaml


@dataclass
class TrainingConfig:
    """Hyperparametry uceni -- predavaji se tride ``Trainer``.

    Atributy
    --------
    lr:
        Krok uceni optimizeru SGD.
    epochs:
        Pocet pruchodu trenovacimi daty.
    batch_size:
        Velikost davky v ``DataLoader``.
    seed:
        Seed inicializace vah (``torch.manual_seed``) i michani davek.
    """

    lr: float
    epochs: int
    batch_size: int
    seed: int


@dataclass
class RegressionConfig:
    """Nastaveni casti A -- regrese (sekce ``regression``).

    Atributy
    --------
    hidden_dims:
        Pocty neuronu skrytych vrstev, napr. ``[32, 16]``.
    test_size:
        Podil testovacich vzorku (nahodne rozdeleni), z intervalu ``(0, 1)``.
    training:
        Hyperparametry uceni teto casti.
    """

    hidden_dims: list[int]
    test_size: float
    training: TrainingConfig


@dataclass
class ForecastConfig:
    """Nastaveni casti B -- predpoved casove rady (sekce ``forecast``).

    Atributy
    --------
    window_size:
        Delka okna: kolik poslednich hodnot rady predpovida dalsi (``>= 1``).
    hidden_dims:
        Pocty neuronu skrytych vrstev; vstup site ma ``window_size`` hodnot.
    train_frac:
        Podil rady na uceni (chronologicke rozdeleni), z intervalu ``(0, 1)``.
    training:
        Hyperparametry uceni teto casti.
    """

    window_size: int
    hidden_dims: list[int]
    train_frac: float
    training: TrainingConfig


@dataclass
class PathsConfig:
    """Cesty ke vstupum a vystupum (sekce ``paths``), relativne ke koreni repozitare.

    Atributy
    --------
    temperature_csv:
        CSV s dennimi minimalnimi teplotami (cast B).
    graphs_dir:
        Slozka pro vystupni grafy (.png).
    models_dir:
        Slozka pro ulozene modely (``state_dict``, .pt).
    logs_dir:
        Slozka s logem uceni (``trainer.log``).
    """

    temperature_csv: str
    graphs_dir: str
    models_dir: str
    logs_dir: str


@dataclass
class SinusoidConfig:
    """Data casti A -- zasumena sinusovka na intervalu ``[-pi, pi]`` (sekce ``sinusoid``).

    Atributy
    --------
    n_samples:
        Pocet bodu (cele cislo ``>= 10``).
    noise:
        Smerodatna odchylka gaussovskeho sumu ve stejnych jednotkach jako ``y``
        (amplituda sinusovky je 1), ``>= 0``.
    """

    n_samples: int
    noise: float


@dataclass
class DiagnosticsConfig:
    """Prahy kontrol, ktere pipeline vypisuje (sekce ``diagnostics``).

    Atributy
    --------
    divergence_factor:
        Uceni divergovalo, kdyz konecna trenovaci ztrata > ``divergence_factor`` x
        ztrata 1. epochy (bezrozmerny nasobek, ``> 1``).
    overfit_rise:
        Preuceni: konecna validacni ztrata je o vic nez tento relativni podil nad
        minimem validacni ztraty (``0.10`` = o 10 %), ``> 0``.
    overfit_min_fraction:
        ... a minimum nastalo pred touto casti epoch (``0.9`` = pred 90 %),
        z intervalu ``(0, 1]``.
    load_tolerance:
        Nejvetsi povoleny rozdil predikci puvodniho a nacteneho modelu
        (ve stejnych jednotkach jako predikce), ``> 0``.
    zero_error_tolerance:
        Baseline s MAE <= tato hodnota se povazuje za nulovou (vraci samotne
        cile), ``>= 0``.
    normality_alpha:
        Hladina vyznamnosti Shapirova-Wilkova testu normality rezidui,
        z intervalu ``(0, 1)``.
    """

    divergence_factor: float
    overfit_rise: float
    overfit_min_fraction: float
    load_tolerance: float
    zero_error_tolerance: float
    normality_alpha: float


@dataclass
class ExperimentConfig:
    """Korenova konfigurace experimentu slozena ze vsech dilcich sekci.

    Atributy
    --------
    paths:
        Cesty ke vstupum a vystupum.
    sinusoid:
        Parametry generovane sinusovky (cast A).
    regression:
        Nastaveni casti A (regrese).
    forecast:
        Nastaveni casti B (casova rada).
    diagnostics:
        Prahy kontrol vypisovanych pipeline.
    seed:
        Spolecny seed (rozdeleni dat, inicializace vah, michani davek), ``>= 0``.
    """

    paths: PathsConfig
    sinusoid: SinusoidConfig
    regression: RegressionConfig
    forecast: ForecastConfig
    diagnostics: DiagnosticsConfig
    seed: int


def _as_float(name: str, value: Any) -> float:
    """Pomocna funkce: prevede hodnotu na ``float``; bool a necisla vyhodi ``ValueError``.

    Retezec se prijme, jen kdyz je to zapis cisla (YAML cte ``1e-3`` jako retezec).
    """
    if isinstance(value, bool):
        raise ValueError(f"{name} musi byt cislo, zadano: {value!r}")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
    raise ValueError(f"{name} musi byt cislo, zadano: {value!r}")


def _as_int(name: str, value: Any) -> int:
    """Pomocna funkce: vrati ``int``; ``float``, ``bool`` a retezec vyhodi ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} musi byt cele cislo, zadano: {value!r}")
    return value


def _as_str(name: str, value: Any) -> str:
    """Pomocna funkce: vrati retezec; cokoli jineho vyhodi ``ValueError``."""
    if not isinstance(value, str):
        raise ValueError(f"{name} musi byt retezec (cesta), zadano: {value!r}")
    return value


def _as_int_list(name: str, value: Any) -> list[int]:
    """Pomocna funkce: vrati seznam celych cisel; jinak ``ValueError``."""
    if not isinstance(value, list):
        raise ValueError(f"{name} musi byt seznam celych cisel, zadano: {value!r}")
    return [_as_int(f"{name}[{i}]", item) for i, item in enumerate(value)]


def _get(section: dict[str, Any], section_name: str | None, key: str) -> Any:
    """Pomocna funkce: vrati ``section[key]``; chybejici klic -> ``ValueError``."""
    if key not in section:
        where = f"v sekci {section_name}" if section_name else "v korenu souboru"
        raise ValueError(f"{where} chybi klic '{key}'")
    return section[key]


def _make_training(section: dict[str, Any], name: str, seed: int) -> TrainingConfig:
    """Pomocna funkce: slozi ``TrainingConfig`` z plochych klicu sekce a seedu."""
    return TrainingConfig(
        lr=_as_float(f"{name}.lr", _get(section, name, "lr")),
        epochs=_as_int(f"{name}.epochs", _get(section, name, "epochs")),
        batch_size=_as_int(f"{name}.batch_size", _get(section, name, "batch_size")),
        seed=seed,
    )


def _get_section(raw: dict[str, Any], name: str, filepath: str) -> dict[str, Any]:
    """Pomocna funkce: vrati sekci ``name``; chybi-li nebo neni slovnik -> ``ValueError``."""
    section = _get(raw, None, name)
    if not isinstance(section, dict):
        raise ValueError(f"{filepath}: sekce '{name}' musi obsahovat klice, zadano: {section!r}")
    return section


def load_config(filepath: str = "config.yaml") -> ExperimentConfig:
    """Nacte a zvaliduje konfiguraci z YAML souboru.

    Parametry
    ---------
    filepath:
        Cesta k YAML souboru s konfiguraci.

    Navratova hodnota
    -----------------
    ``ExperimentConfig`` s vnorenymi dataclassami ``RegressionConfig``,
    ``ForecastConfig`` (kazda s ``TrainingConfig``) a korenovym ``seed``.

    Vyjimky
    -------
    ``FileNotFoundError``:
        Pokud soubor neexistuje.
    ``ValueError``:
        Pokud YAML nelze naparsovat, chybi sekce/klic, klic ma spatny typ,
        nebo hodnota nesplnuje kontroly ve ``validate_config``.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as handle:
            raw: Any = yaml.safe_load(handle)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Konfiguracni soubor '{filepath}' neexistuje (spoustejte skript z korene repozitare)"
        ) from exc
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        line = f" na radku {mark.line + 1}" if mark is not None else ""
        problem = getattr(exc, "problem", None) or str(exc)
        raise ValueError(
            f"{filepath} neni platny YAML{line}: {problem}. "
            "Pozor, v odsazeni nesmi byt tabulatory, jen mezery."
        ) from exc
    except UnicodeDecodeError as exc:
        raise ValueError(f"{filepath} nelze precist jako text UTF-8: {exc.reason}") from exc

    if not isinstance(raw, dict):
        raise ValueError(
            f"{filepath} nema ocekavanou strukturu: koren musi obsahovat sekce "
            f"'regression' a 'forecast' a klic 'seed' (nacteno: {raw!r})"
        )

    seed = _as_int("seed", _get(raw, None, "seed"))
    paths = _get_section(raw, "paths", filepath)
    sin = _get_section(raw, "sinusoid", filepath)
    reg = _get_section(raw, "regression", filepath)
    fc = _get_section(raw, "forecast", filepath)
    diag = _get_section(raw, "diagnostics", filepath)
    cfg = ExperimentConfig(
        paths=PathsConfig(
            **{k: _as_str(f"paths.{k}", _get(paths, "paths", k))
               for k in ("temperature_csv", "graphs_dir", "models_dir", "logs_dir")}
        ),
        sinusoid=SinusoidConfig(
            n_samples=_as_int("sinusoid.n_samples", _get(sin, "sinusoid", "n_samples")),
            noise=_as_float("sinusoid.noise", _get(sin, "sinusoid", "noise")),
        ),
        regression=RegressionConfig(
            hidden_dims=_as_int_list(
                "regression.hidden_dims", _get(reg, "regression", "hidden_dims"),
            ),
            test_size=_as_float("regression.test_size", _get(reg, "regression", "test_size")),
            training=_make_training(reg, "regression", seed),
        ),
        forecast=ForecastConfig(
            window_size=_as_int("forecast.window_size", _get(fc, "forecast", "window_size")),
            hidden_dims=_as_int_list("forecast.hidden_dims", _get(fc, "forecast", "hidden_dims")),
            train_frac=_as_float("forecast.train_frac", _get(fc, "forecast", "train_frac")),
            training=_make_training(fc, "forecast", seed),
        ),
        diagnostics=DiagnosticsConfig(
            **{k: _as_float(f"diagnostics.{k}", _get(diag, "diagnostics", k))
               for k in ("divergence_factor", "overfit_rise", "overfit_min_fraction",
                         "load_tolerance", "zero_error_tolerance", "normality_alpha")}
        ),
        seed=seed,
    )
    validate_config(cfg)
    return cfg


def _validate_training(name: str, tr: TrainingConfig) -> None:
    """Pomocna funkce: zkontroluje hyperparametry uceni sekce ``name``."""
    if not math.isfinite(tr.lr) or tr.lr <= 0:
        raise ValueError(f"{name}.lr musi byt konecne cislo > 0, zadano: {tr.lr}")
    if tr.epochs < 1:
        raise ValueError(f"{name}.epochs musi byt >= 1, zadano: {tr.epochs}")
    if tr.batch_size < 1:
        raise ValueError(f"{name}.batch_size musi byt >= 1, zadano: {tr.batch_size}")
    _validate_seed(f"{name}.seed", tr.seed)


def _validate_seed(name: str, seed: Any) -> None:
    """Pomocna funkce: seed musi byt cele cislo >= 0."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"{name} musi byt cele cislo >= 0, zadano: {seed!r}")


def _validate_hidden_dims(name: str, hidden_dims: list[int]) -> None:
    """Pomocna funkce: ``hidden_dims`` musi byt neprazdny seznam cisel >= 1."""
    if len(hidden_dims) == 0 or any(h < 1 for h in hidden_dims):
        raise ValueError(
            f"{name}.hidden_dims musi byt neprazdny seznam celych cisel >= 1, "
            f"zadano: {hidden_dims}"
        )


def validate_config(cfg: ExperimentConfig) -> None:
    """Zkontroluje rozsahy hodnot v konfiguraci.

    Pri poruseni nektere podminky vyhodi ``ValueError`` se srozumitelnou
    ceskou hlaskou obsahujici jmeno klice a zadanou hodnotu. Kontroluji se
    (pro obe sekce ``regression`` a ``forecast``):

    - ``seed`` cele cislo ``>= 0`` (korenovy i v ``TrainingConfig``)
    - ``lr`` konecne a ``> 0``, ``epochs >= 1``, ``batch_size >= 1``
    - ``hidden_dims`` neprazdny a kazda polozka ``>= 1``
    - ``regression.test_size`` v intervalu ``(0, 1)``
    - ``forecast.window_size >= 1``
    - ``forecast.train_frac`` v intervalu ``(0, 1)``

    A dale:

    - ``paths.*`` neprazdne retezce
    - ``sinusoid.n_samples`` cele cislo ``>= 10``, ``sinusoid.noise`` konecne ``>= 0``
    - ``diagnostics.*`` vsechna konecna cisla; ``divergence_factor > 1``,
      ``overfit_rise > 0``, ``0 < overfit_min_fraction <= 1``, ``load_tolerance > 0``,
      ``zero_error_tolerance >= 0``, ``0 < normality_alpha < 1``

    Navratova hodnota je ``None`` -- funkce pouze validuje.
    """
    _validate_seed("seed", cfg.seed)

    reg = cfg.regression
    _validate_training("regression", reg.training)
    _validate_hidden_dims("regression", reg.hidden_dims)
    if not 0.0 < reg.test_size < 1.0:
        raise ValueError(
            f"regression.test_size musi lezet v intervalu (0, 1), zadano: {reg.test_size}"
        )

    fc = cfg.forecast
    _validate_training("forecast", fc.training)
    _validate_hidden_dims("forecast", fc.hidden_dims)
    if fc.window_size < 1:
        raise ValueError(f"forecast.window_size musi byt >= 1, zadano: {fc.window_size}")
    if not 0.0 < fc.train_frac < 1.0:
        raise ValueError(
            f"forecast.train_frac musi lezet v intervalu (0, 1), zadano: {fc.train_frac}"
        )

    for key in ("temperature_csv", "graphs_dir", "models_dir", "logs_dir"):
        value = getattr(cfg.paths, key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"paths.{key} musi byt neprazdny retezec, zadano: {value!r}")

    sin = cfg.sinusoid
    if isinstance(sin.n_samples, bool) or not isinstance(sin.n_samples, int) \
            or sin.n_samples < 10:
        raise ValueError(f"sinusoid.n_samples musi byt cele cislo >= 10, zadano: {sin.n_samples!r}")
    if not math.isfinite(sin.noise) or sin.noise < 0:
        raise ValueError(f"sinusoid.noise musi byt konecne cislo >= 0, zadano: {sin.noise}")

    diag = cfg.diagnostics
    for key in ("divergence_factor", "overfit_rise", "overfit_min_fraction", "load_tolerance",
                "zero_error_tolerance", "normality_alpha"):
        value = getattr(diag, key)
        if not math.isfinite(value):
            raise ValueError(f"diagnostics.{key} musi byt konecne cislo, zadano: {value}")
    if diag.divergence_factor <= 1:
        raise ValueError(
            f"diagnostics.divergence_factor musi byt > 1, zadano: {diag.divergence_factor}"
        )
    if diag.overfit_rise <= 0:
        raise ValueError(f"diagnostics.overfit_rise musi byt > 0, zadano: {diag.overfit_rise}")
    if not 0.0 < diag.overfit_min_fraction <= 1.0:
        raise ValueError(
            "diagnostics.overfit_min_fraction musi lezet v intervalu (0, 1], "
            f"zadano: {diag.overfit_min_fraction}"
        )
    if diag.load_tolerance <= 0:
        raise ValueError(f"diagnostics.load_tolerance musi byt > 0, zadano: {diag.load_tolerance}")
    if diag.zero_error_tolerance < 0:
        raise ValueError(
            f"diagnostics.zero_error_tolerance musi byt >= 0, zadano: {diag.zero_error_tolerance}"
        )
    if not 0.0 < diag.normality_alpha < 1.0:
        raise ValueError(
            f"diagnostics.normality_alpha musi lezet v intervalu (0, 1), "
            f"zadano: {diag.normality_alpha}"
        )
