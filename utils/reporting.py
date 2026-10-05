"""Hlasky pipeline -- jednotne vypisy do konzole (PREDVYPLNENO).

Jde o orchestraci pipeline ``cviceni_11.py``, kterou student neupravuje.
Lezi mimo ``src/`` (studentske ukoly) i ``dataio/`` (data, konfigurace,
grafy), protoze nepatri k zadne z techto vrstev. Smer zavislosti je
``utils`` -> ``src``, ``dataio`` a nikdy naopak: ``src/`` ani ``dataio/``
o ``utils/`` nevi.

Vsechny hlasky maji pevne znacky (``[NENI HOTOVO]``, ``[CHYBA IMPLEMENTACE]``,
``[CHYBA DAT]``, ``[PRESKOCENO]``), podle kterych se da vystup pipeline
snadno prohledat.
"""

from __future__ import annotations


def banner(text: str) -> None:
    """Vypise oddelovaci nadpis faze pipeline.

    Parametry
    ---------
    text : str
        Text nadpisu.
    """
    print("\n" + "=" * 78)
    print(f"  {text}")
    print("=" * 78)


def faze_neni_hotova(exc: NotImplementedError) -> None:
    """Vypise pratelskou hlasku, kdyz faze narazi na nedokonceny ukol.

    Parametry
    ---------
    exc : NotImplementedError
        Vyjimka s textem ``Úkol: ...``.
    """
    print(f"  [NENI HOTOVO] {exc}")
    print("  -> Tuto cast dokoncite v ramci ukolu; pipeline pokracuje dal.")


def chyba_implementace(exc: Exception) -> None:
    """Vypise hlasku, kdyz dokoncena cast selze (assert, tvar, preklep, spatny index).

    Parametry
    ---------
    exc : Exception
        Zachycena vyjimka.
    """
    print(f"  [CHYBA IMPLEMENTACE] {type(exc).__name__}: {exc}")
    print("  -> Zkontrolujte svuj kod (tvary tenzoru, asserty, indexy, jmena atributu; "
          "README, Pokyny k vypracovani); pipeline pokracuje dal.")


def chyba_dat(exc: Exception) -> None:
    """Vypise hlasku o chybe vstupnich dat (chybejici, prazdny ci poskozeny soubor).

    Parametry
    ---------
    exc : Exception
        Zachycena vyjimka; u ``FileNotFoundError`` a ``ValueError`` se vypise jen
        jeji (ceska) hlaska, u ostatnich i jmeno typu.
    """
    if isinstance(exc, (FileNotFoundError, ValueError)):
        print(f"  [CHYBA DAT] {exc}")
    else:
        print(f"  [CHYBA DAT] {type(exc).__name__}: {exc}")


def preskoceno(duvod: str) -> None:
    """Vypise hlasku o preskocene fazi, jejiz vstup se nepodarilo ziskat.

    Parametry
    ---------
    duvod : str
        Proc se faze preskakuje.
    """
    print(f"  [PRESKOCENO] {duvod}")
