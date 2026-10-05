"""Nacitani a generovani dat pro cviceni 11 -- vse PREDVYPLNENE.

Cast A (regrese, nezavisle vzorky):

* ``make_sinusoid`` -- zasumena sinusovka, 1D regrese.
* ``load_diabetes_data`` -- tabulkova regrese (10 priznaku), standardizovana.
* ``random_split`` -- NAHODNE rozdeleni na trenovaci a testovaci cast.

Cast B (casova rada):

* ``load_temperature_series`` -- denni minimalni teploty (Melbourne, 1981-1990).
* ``chronological_split`` -- rozdeleni rady BEZ michani (zacatek = uceni, konec = test).
"""

from __future__ import annotations

import csv
import io
import math
import os
import re

import numpy as np

TEMPERATURE_CSV = "data/daily_min_temperatures.csv"

_DATE_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}")
_EXPECTED_FORMAT = (
    "datum,teplota s desetinnou teckou, oddelovac carka (napr. 1981-01-01,20.7)"
)


def random_split(
    x: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int | None = 42,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Nahodne rozdeli data na trenovaci a testovaci cast.

    Michani je zde v poradku, protoze vzorky jsou **nezavisle** (kazdy pacient
    je samostatny pripad, poradi radku nic neznamena). Pro casovou radu to
    NEPLATI -- tam se pouziva ``chronological_split``.

    Parametry
    ---------
    x:
        Priznaky tvaru ``(n, d)`` nebo ``(n,)``.
    y:
        Cil tvaru ``(n,)``.
    test_size:
        Podil testovacich vzorku z intervalu ``(0, 1)``,
        ``n_test = max(1, round(n * test_size))``.
    random_state:
        Seed generatoru nahodnych cisel (reprodukovatelnost).

    Navratova hodnota
    -----------------
    ``(x_train, y_train, x_test, y_test)``.

    Vyjimky
    -------
    ``ValueError``:
        Pokud ``x`` a ``y`` nemaji stejny pocet vzorku, ``test_size`` neni
        v ``(0, 1)`` nebo by trenovaci cast vysla prazdna.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    if len(y) != n:
        raise ValueError(f"x a y musi mit stejny pocet vzorku, zadano: {n} a {len(y)}")
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size musi lezet v intervalu (0, 1), zadano: {test_size}")
    rng = np.random.default_rng(random_state)
    permutation = rng.permutation(n)
    n_test = max(1, round(n * test_size))
    if n_test >= n:
        raise ValueError(
            f"test_size {test_size} pro {n} vzorku nechava prazdnou trenovaci cast"
        )
    test_idx = permutation[:n_test]
    train_idx = permutation[n_test:]
    return x[train_idx], y[train_idx], x[test_idx], y[test_idx]


def load_diabetes_data(
    test_size: float = 0.2,
    random_state: int | None = 42,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Nacte dataset diabetes, rozdeli ho a standardizuje priznaky i cil.

    Parametry
    ---------
    test_size:
        Podil testovacich vzorku, vychozi ``0.2``.
    random_state:
        Seed nahodneho rozdeleni.

    Navratova hodnota
    -----------------
    ``(x_train, y_train, x_test, y_test, y_mean, y_std)``:

    * ``x_*`` tvaru ``(n, 10)``, ``float64``, standardizovane,
    * ``y_*`` tvaru ``(n,)``, ``float64``, **standardizovane**,
    * ``y_mean``, ``y_std`` -- prumer a odchylka cile z trenovaci casti.

    Predikce site se do puvodnich jednotek prevedou jako
    ``y_pred * y_std + y_mean``.

    Vyjimky
    -------
    ``ValueError``:
        Pokud ma trenovaci cast mene nez 2 vzorky nebo nektery priznak ci cil
        ma v ni nulovou odchylku (standardizace by dala same NaN).

    Priznaky
    --------
    ``age`` (vek), ``sex`` (pohlavi), ``bmi`` (index telesne hmotnosti, index 2),
    ``bp`` (stredni krevni tlak), ``s1`` az ``s6`` (hodnoty krevniho sera).
    Cil je mira progrese onemocneni rok po vstupnim vysetreni (cca 25 az 346).

    Standardizace
    -------------
    Prumer a odchylka se pocitaji **jen z trenovaci casti** a stejne se pouziji
    na testovaci cast -- jinak by informace z testu "unikla" do uceni (leakage,
    viz cviceni 6). Standardizuje se i cil: ma prumer kolem 150, a s krokem
    uceni 0.01 a aktivaci tanh by to vedlo k obrim gradientum a nestabilnimu
    uceni. Po standardizaci ma cil prumer ~0 a odchylku ~1.
    """
    from sklearn.datasets import load_diabetes

    dataset = load_diabetes(scaled=False)
    x = np.asarray(dataset.data, dtype=np.float64)
    y = np.asarray(dataset.target, dtype=np.float64)

    x_train, y_train, x_test, y_test = random_split(x, y, test_size, random_state)
    if len(x_train) < 2:
        raise ValueError(
            f"Trenovaci cast ma jen {len(x_train)} vzorek; pro standardizaci jsou "
            f"potreba aspon 2 (test_size = {test_size})"
        )

    x_mean = x_train.mean(axis=0)
    x_std = x_train.std(axis=0)
    y_mean = float(y_train.mean())
    y_std = float(y_train.std())
    if np.any(x_std == 0) or y_std == 0:
        raise ValueError(
            "Nektery priznak nebo cil ma v trenovaci casti nulovou odchylku, "
            f"standardizace nelze provest (test_size = {test_size})"
        )

    x_train = (x_train - x_mean) / x_std
    x_test = (x_test - x_mean) / x_std
    y_train = (y_train - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std
    return x_train, y_train, x_test, y_test, y_mean, y_std


def make_sinusoid(
    n: int = 300,
    noise: float = 0.1,
    seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray]:
    """Vygeneruje zasumenou sinusovku pro 1D regresi.

    Parametry
    ---------
    n:
        Pocet bodu.
    noise:
        Smerodatna odchylka gaussovskeho sumu.
    seed:
        Seed generatoru (reprodukovatelnost).

    Navratova hodnota
    -----------------
    ``(x, y)`` tvaru ``(n,)``, ``float64``; ``x`` je serazene v ``[-pi, pi]``,
    ``y = sin(x) + sum``.
    """
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(-np.pi, np.pi, n))
    y = np.sin(x) + rng.normal(0.0, noise, n)
    return x, y


def _is_number(text: str) -> bool:
    """Pomocna funkce: je ``text`` (po odstraneni uvodniho ``?``) konecne cislo?"""
    try:
        return math.isfinite(float(text.strip().removeprefix("?")))
    except ValueError:
        return False


def _bad_row(path: str, line_no: int, row: list[str], problem: str) -> ValueError:
    """Pomocna funkce: sestavi ``ValueError`` s ceskou hlaskou o spatnem radku."""
    content = ",".join(row)
    if len(content) > 60:
        content = content[:57] + "..."
    return ValueError(
        f"Soubor '{path}', radek {line_no}: {problem}. Obsah radku: {content!r}. "
        f"Ocekavany format: {_EXPECTED_FORMAT}."
    )


def load_temperature_series(path: str = TEMPERATURE_CSV) -> np.ndarray:
    """Nacte denni minimalni teploty (Melbourne, 1981-1990) jako 1D pole.

    Parametry
    ---------
    path:
        Cesta k CSV souboru, vychozi ``TEMPERATURE_CSV``.

    Navratova hodnota
    -----------------
    Zapisovatelne 1D pole ``float64`` o 3650 hodnotach v poradi dnu
    (prvni 20.7, posledni 13.0).

    Vyjimky
    -------
    ``FileNotFoundError``:
        Pokud soubor neexistuje.
    ``ValueError``:
        Pokud soubor nelze precist (prazdny, spatne kodovani, useknuty) nebo
        nema ocekavany format. Hlaska obsahuje cestu, cislo a obsah radku.

    Format
    ------
    Soubor je puvodni export z DataMarketu s temito zvlastnostmi:

    * nazev sloupce v hlavicce obsahuje carku (je v uvozovkach),
    * tri hodnoty maji prefix ``?`` (artefakt exportu, napr. ``?0.2``),
    * za daty je prazdny radek a textova paticka.

    Pravidla nacitani (nic se neprecte "potichu"): prvni neprazdny radek je
    hlavicka, pokud nezacina datem a jeho posledni pole neni cislo (soubor bez
    hlavicky tedy neztrati prvni den); prazdne radky se preskoci; kazdy datovy
    radek ``RRRR-MM-DD,hodnota`` ma presne 2 pole a konecne cislo (uvodni ``?``
    se odstrani); radek, ktery nezacina datem, je povolen jen jako paticka, tj.
    uz za nim nesmi nasledovat zadny datovy radek. Funguje i pro cisty CSV
    ``Date,Temp`` (i s BOM a CRLF).
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Soubor s teplotami '{path}' neexistuje. Ulozte ho tam (vychozi cesta je "
            f"{TEMPERATURE_CSV}) a spoustejte skript z korene repozitare."
        )
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as handle:
            text = handle.read()
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"Soubor '{path}' nelze precist jako text UTF-8 ({exc.reason}); "
            f"ocekavany format: {_EXPECTED_FORMAT}."
        ) from exc
    except OSError as exc:
        raise ValueError(f"Soubor '{path}' se nepodarilo precist: {exc}") from exc

    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    values: list[float] = []
    footer: tuple[int, list[str]] | None = None   # prvni radek mimo data za zacatkem dat
    first_row: tuple[int, list[str]] | None = None   # prvni neprazdny radek (pro hlasky)
    seen_first = False
    try:
        for row in reader:
            line_no = reader.line_num
            if not row or all(not cell.strip() for cell in row):
                continue
            if first_row is None:
                first_row = (line_no, row)
            first = row[0].strip()
            is_date = _DATE_PATTERN.fullmatch(first) is not None
            if not seen_first:
                seen_first = True
                if not is_date and not _is_number(row[-1]):
                    continue                          # hlavicka
            if not is_date:
                if footer is None:
                    footer = (line_no, row)
                continue
            if footer is not None:
                raise _bad_row(path, footer[0], footer[1],
                               "radek bez data uprostred dat (paticka smi byt jen na konci)")
            if len(row) != 2:
                raise _bad_row(path, line_no, row,
                               f"datovy radek ma {len(row)} poli misto 2")
            raw = row[1].strip().removeprefix("?")
            try:
                value = float(raw)
            except ValueError:
                raise _bad_row(path, line_no, row, "hodnota teploty neni cislo") from None
            if not math.isfinite(value):
                raise _bad_row(path, line_no, row, "hodnota teploty neni konecne cislo")
            values.append(value)
    except csv.Error as exc:
        raise ValueError(
            f"Soubor '{path}', radek {reader.line_num}: chyba CSV ({exc}); soubor je "
            f"poskozeny nebo useknuty. Ocekavany format: {_EXPECTED_FORMAT}."
        ) from exc

    if not values:
        if first_row is None:
            raise ValueError(f"Soubor '{path}' je prazdny. Ocekavany format: {_EXPECTED_FORMAT}.")
        raise _bad_row(path, first_row[0], first_row[1],
                       "soubor neobsahuje zadne datove radky (radek nezacina datem "
                       "RRRR-MM-DD nebo je spatny oddelovac)")
    return np.array(values, dtype=np.float64)


def chronological_split(
    series: np.ndarray,
    train_frac: float = 0.8,
    ) -> tuple[np.ndarray, np.ndarray]:
    """Rozdeli casovou radu na zacatek (uceni) a konec (test) BEZ michani.

    Parametry
    ---------
    series:
        Casova rada (1D pole) v chronologickem poradi.
    train_frac:
        Podil rady urceny k uceni z intervalu ``(0, 1)``, vychozi ``0.8``.

    Navratova hodnota
    -----------------
    ``(train, test)`` -- kopie ``series[:n_train]`` a ``series[n_train:]``,
    ``n_train = int(len(series) * train_frac)``.

    Vyjimky
    -------
    ``ValueError``:
        Pokud ``train_frac`` neni v ``(0, 1)`` nebo by nektera cast vysla prazdna.

    Proc se nemicha
    ---------------
    Hodnoty casove rady NEJSOU nezavisle: zitrek navazuje na dnesek. Kdybychom
    radu zamichali (jako ``random_split``), model by se ucil i na dnech, ktere
    lezi PO testovanych dnech, tedy by "videl budoucnost". To je unik dat
    (leakage, obdoba z cviceni 6) a testovaci chyba by byla nerealisticky
    dobra. Realne se predpovida budoucnost z minulosti, takze test musi lezet
    za uceni v case.
    """
    if not 0.0 < train_frac < 1.0:
        raise ValueError(f"train_frac musi lezet v intervalu (0, 1), zadano: {train_frac}")
    series = np.asarray(series)
    n_train = int(len(series) * train_frac)
    if n_train == 0 or n_train == len(series):
        raise ValueError(
            f"Rozdeleni rady delky {len(series)} s train_frac = {train_frac} "
            "by vytvorilo prazdnou cast"
        )
    return series[:n_train].copy(), series[n_train:].copy()
