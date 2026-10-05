# -*- coding: utf-8 -*-

"""
Created on 02. 10. 2026 at 14:00:00

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
    Vstupni bod cviceni 11 (regrese a predpoved neuronovou siti v PyTorch).
    Pipeline ma dve casti a deset fazi.

    Cast A -- regrese (nezavisle vzorky, NAHODNE rozdeleni):
      A1. data -- zasumena sinusovka a diabetes (10 priznaku, standardizace),
      A2. sinusovka -- uceni site, prolozeni krivky, krivka uceni, MSE/MAE/R^2,
      A3. diabetes -- uceni site, krivka uceni,
      A4. diabetes -- MSE, MAE a R^2 na testovacich datech v puvodnich jednotkach,
      A5. diabetes -- rezidualni diagnostika (4 grafy, Shapiruv-Wilkuv test),
      A6. diabetes -- ulozeni state_dict do models/ a kontrolni nacteni.

    Cast B -- predpoved casove rady (denni minimalni teploty, Melbourne):
      B1. data -- nacteni rady, CHRONOLOGICKE rozdeleni, standardizace,
      B2. okenkovani (WindowedDataset) a uceni site, ulozeni modelu,
      B3. MAE a RMSE modelu proti persistence baseline ("zitra jako dnes"),
      B4. grafy predpovedi v case a rezidui v case.

    Vse nastavitelne (cesty, sinusovka, hyperparametry, prahy kontrol) je
    v config.yaml; tento soubor neobsahuje zadne konstanty. Pomocny kod
    pipeline (hlasky, prepravky na vysledky fazi, spolecne uceni a kontrola
    jeho prubehu) je v balicku utils/.

    Cela pipeline je predvyplnena. Studentske ukoly jsou v src/ (MLP,
    Trainer._train_step, Trainer.save/load, WindowedDataset, metriky,
    persistence_baseline) -- viz README, Pokyny k vypracovani.

    Repozitar bezi v kazdem stavu. Dokud nejsou ukoly hotove, faze se
    zastavi jen hlaskou [NENI HOTOVO] Úkol: ... a pipeline pokracuje dal;
    faze zavisla na neuspesne fazi se ohlasi jako [PRESKOCENO]. Jakakoli
    jina vyjimka z faze je [CHYBA IMPLEMENTACE], chyba dat [CHYBA DAT],
    divergence uceni [CHYBA UCENI]. Nikdy nezpracovany traceback.
================================================================================
"""

from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import torch

# --- Import guard: srozumitelna hlaska misto holeho ImportError ----------------
try:
    from dataio import (
        ExperimentConfig,
        chronological_split,
        load_config,
        load_diabetes_data,
        load_temperature_series,
        make_sinusoid,
        plot_forecast,
        plot_loss_curve,
        plot_predicted_vs_actual,
        plot_regression_fit,
        plot_residuals_in_time,
        random_split,
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
    from utils import (
        DiabetesData,
        DiabetesModel,
        ForecastModel,
        SeriesData,
        SinusData,
        UceniDivergovalo,
        banner,
        chyba_dat,
        chyba_implementace,
        faze_neni_hotova,
        nauc_model,
        predikuj,
        preskoceno,
        vstupni_matice,
    )
except ImportError as exc:  # pragma: no cover - jen ochranna hlaska
    print(f"[CHYBA IMPORTU] Nepodarilo se nacist moduly projektu: {exc}")
    print("Zkontrolujte, ze spoustite skript z korene repozitare a mate "
          "nainstalovane zavislosti (pip install -r requirements.txt).")
    sys.exit(1)


# ============================================================================ #
#  Cast A: regrese                                                             #
# ============================================================================ #
def faze_a1_data(cfg: ExperimentConfig) -> tuple[SinusData, DiabetesData] | None:
    """Faze A1 -- sinusovka a diabetes: generovani, rozdeleni, standardizace.

    Navratova hodnota
    -----------------
    tuple[SinusData, DiabetesData] | None
        Data, nebo ``None`` pri datove chybe (cela cast A se pak preskoci).
    """
    banner("Faze A1: Data -- sinusovka a diabetes (nahodne rozdeleni)")
    try:
        x, y = make_sinusoid(n=cfg.sinusoid.n_samples, noise=cfg.sinusoid.noise, seed=cfg.seed)
        x_tr, y_tr, x_te, y_te = random_split(
            x, y, test_size=cfg.regression.test_size, random_state=cfg.seed,
        )
        sinus = SinusData(x, y, x_tr, y_tr, x_te, y_te)
        print(f"  sinusovka:  trenink x {x_tr.shape}, y {y_tr.shape}; "
              f"test x {x_te.shape}, y {y_te.shape}")

        xd_tr, yd_tr, xd_te, yd_te, y_mean, y_std = load_diabetes_data(
            test_size=cfg.regression.test_size, random_state=cfg.seed,
        )
    except Exception as exc:
        chyba_dat(exc)
        print("  -> Zkontrolujte regression.test_size a sinusoid.n_samples v config.yaml; "
              "cela cast A se preskoci, cast B pokracuje.")
        return None
    diabetes = DiabetesData(xd_tr, yd_tr, xd_te, yd_te, y_mean, y_std)
    print(f"  diabetes:   trenink x {xd_tr.shape}, y {yd_tr.shape}; "
          f"test x {xd_te.shape}, y {yd_te.shape}")
    print(f"  diabetes: standardizace z trenovacich dat; cil mel prumer {y_mean:.1f} "
          f"a odchylku {y_std:.1f} (predikce se prevadi zpet: y * y_std + y_mean)")
    return sinus, diabetes


def faze_a2_sinusovka(cfg: ExperimentConfig, sinus: SinusData | None) -> Trainer | None:
    """Faze A2 -- sinusovka: uceni, prolozeni krivky, krivka uceni a metriky.

    Vypocet metrik je ve vlastnim ``try``, aby nedokoncene metriky nezahodily
    hotove uceni a grafy.

    Parametry
    ---------
    cfg : ExperimentConfig
        Konfigurace experimentu.
    sinus : SinusData | None
        Data z faze A1 (``None`` = faze A1 selhala).

    Navratova hodnota
    -----------------
    Trainer | None
        Naucena sit, nebo ``None``, kdyz uceni selhalo.
    """
    banner("Faze A2: Sinusovka -- uceni site a prolozeni krivky")
    if sinus is None:
        preskoceno("Data se nepodarilo pripravit (faze A1 selhala).")
        return None
    reg = cfg.regression
    graphs = cfg.paths.graphs_dir
    print(f"  architektura 1-{'-'.join(str(h) for h in reg.hidden_dims)}-1 "
          f"(tanh, linearni vystup); lr {reg.training.lr}, epoch {reg.training.epochs}")
    try:
        torch.manual_seed(cfg.seed)                    # seed PRED vytvorenim modelu
        model = MLP(1, reg.hidden_dims)
        train_ds = TabularDataset(sinus.x_train, sinus.y_train)
        test_ds = TabularDataset(sinus.x_test, sinus.y_test)
        trainer, history = nauc_model(
            model, train_ds, test_ds, reg.training, cfg.paths.logs_dir, cfg.diagnostics,
            section="regression",
        )

        x_grid = np.linspace(-np.pi, np.pi, 200)
        grid_pred = trainer.predict(
            torch.as_tensor(x_grid, dtype=torch.float32).reshape(-1, 1)
        ).detach().numpy().reshape(-1)
        fit_png = os.path.join(graphs, "sinus_prolozeni.png")
        curve_png = os.path.join(graphs, "sinus_krivka_uceni.png")
        plot_regression_fit(sinus.x, sinus.y, x_grid, grid_pred, save_path=fit_png)
        plot_loss_curve(history, save_path=curve_png, title="Sinusovka: krivka uceni")
        print(f"  grafy ulozeny: {fit_png}, {curve_png}")
    except NotImplementedError as exc:
        faze_neni_hotova(exc)
        return None
    except UceniDivergovalo as exc:
        print(f"  [CHYBA UCENI] {exc} (regression.lr)")
        return None
    except Exception as exc:
        chyba_implementace(exc)
        return None

    # Metriky zvlast: jejich pripadna nedokoncenost nesmi zahodit hotove uceni.
    try:
        y_pred = predikuj(trainer, test_ds)
        y_true = sinus.y_test
        print(f"  test: MSE {mse(y_true, y_pred):.4f}, MAE {mae(y_true, y_pred):.4f}, "
              f"R^2 {r2(y_true, y_pred):.4f}")
    except NotImplementedError as exc:
        faze_neni_hotova(exc)
    except Exception as exc:
        chyba_implementace(exc)
    return trainer


def faze_a3_diabetes_uceni(
    cfg: ExperimentConfig, diabetes: DiabetesData | None,
    ) -> tuple[DiabetesModel | None, str]:
    """Faze A3 -- diabetes: uceni site a krivka uceni.

    Parametry
    ---------
    cfg : ExperimentConfig
        Konfigurace experimentu.
    diabetes : DiabetesData | None
        Data z faze A1 (``None`` = faze A1 selhala).

    Navratova hodnota
    -----------------
    tuple[DiabetesModel | None, str]
        Trener a predikce na testu v puvodnich jednotkach (nebo ``None``) a duvod
        pro faze, ktere na tomto vysledku zavisi.
    """
    banner("Faze A3: Diabetes -- uceni site (10 priznaku -> progrese)")
    if diabetes is None:
        duvod = "Data se nepodarilo pripravit (faze A1 selhala)."
        preskoceno(duvod)
        return None, duvod
    reg = cfg.regression
    duvod = "Sit nebyla naucena (faze A3 neprobehla)."
    print(f"  architektura 10-{'-'.join(str(h) for h in reg.hidden_dims)}-1; "
          f"lr {reg.training.lr}, epoch {reg.training.epochs}")
    try:
        torch.manual_seed(cfg.seed)
        model = MLP(diabetes.x_train.shape[1], reg.hidden_dims)
        train_ds = TabularDataset(diabetes.x_train, diabetes.y_train)
        test_ds = TabularDataset(diabetes.x_test, diabetes.y_test)
        trainer, history = nauc_model(
            model, train_ds, test_ds, reg.training, cfg.paths.logs_dir, cfg.diagnostics,
            section="regression",
        )
        curve_png = os.path.join(cfg.paths.graphs_dir, "diabetes_krivka_uceni.png")
        plot_loss_curve(history, save_path=curve_png,
                        title="Diabetes: krivka uceni (standardizovany cil)")
        print(f"  graf ulozen: {curve_png}")
        y_pred_std = predikuj(trainer, test_ds)
        y_pred = y_pred_std * diabetes.y_std + diabetes.y_mean
        return DiabetesModel(trainer, y_pred), duvod
    except NotImplementedError as exc:
        faze_neni_hotova(exc)
    except UceniDivergovalo as exc:
        print(f"  [CHYBA UCENI] {exc} (regression.lr)")
        duvod = "Uceni v fazi A3 divergovalo -- zmensete regression.lr v config.yaml."
    except Exception as exc:
        chyba_implementace(exc)
    return None, duvod


def faze_a4_diabetes_metriky(
    cfg: ExperimentConfig, diabetes: DiabetesData | None,
    vysledek: DiabetesModel | None, duvod: str,
    ) -> None:
    """Faze A4 -- diabetes: MSE, MAE a R^2 na testu v puvodnich jednotkach.

    Graf predikce proti skutecnosti se kresli ve vlastnim ``try``, nezavisle
    na metrikach (ktere mohou byt nedokoncene).
    """
    banner("Faze A4: Diabetes -- metriky na testovacich datech")
    if vysledek is None or diabetes is None:
        preskoceno(duvod)
        return
    y_true = diabetes.y_test * diabetes.y_std + diabetes.y_mean
    y_pred = vysledek.y_pred_test
    try:
        hodnota_mse = mse(y_true, y_pred)
        hodnota_mae = mae(y_true, y_pred)
        hodnota_r2 = r2(y_true, y_pred)
        print(f"  MSE   {hodnota_mse:10.2f}   (cil je v puvodnich jednotkach progrese)")
        print(f"  RMSE  {hodnota_mse ** 0.5:10.2f}")
        print(f"  MAE   {hodnota_mae:10.2f}   (typicka chyba predpovedi)")
        print(f"  R^2   {hodnota_r2:10.3f}")
        if hodnota_r2 < 0:
            print(f"  Interpretace: R^2 = {hodnota_r2:.2f} < 0, model je horsi nez "
                  "predpoved prumerem.")
        else:
            print(f"  Interpretace: R^2 = {hodnota_r2:.2f} znamena, ze model na testovacich "
                  f"datech vysvetluje priblizne {100 * hodnota_r2:.0f} % rozptylu progrese "
                  "(R^2 = 0 by odpovidalo predpovedi prumerem).")
    except NotImplementedError as exc:
        faze_neni_hotova(exc)
    except Exception as exc:
        chyba_implementace(exc)
    try:
        pva_png = os.path.join(cfg.paths.graphs_dir, "diabetes_predikce_vs_skutecnost.png")
        plot_predicted_vs_actual(y_true, y_pred, save_path=pva_png)
        print(f"  graf ulozen: {pva_png}")
    except Exception as exc:
        chyba_implementace(exc)


def faze_a5_diabetes_rezidua(
    cfg: ExperimentConfig, diabetes: DiabetesData | None,
    vysledek: DiabetesModel | None, duvod: str,
    ) -> None:
    """Faze A5 -- diabetes: rezidualni diagnostika (ctyri grafy + Shapiruv-Wilkuv test)."""
    banner("Faze A5: Diabetes -- rezidualni diagnostika")
    if vysledek is None or diabetes is None:
        preskoceno(duvod)
        return
    graphs = cfg.paths.graphs_dir
    alpha = cfg.diagnostics.normality_alpha
    try:
        y_true = diabetes.y_test * diabetes.y_std + diabetes.y_mean
        analyzer = ResidualAnalyzer(y_true, vysledek.y_pred_test)
        bmi = diabetes.x_test[:, 2]                    # sloupec 2 = bmi (standardizovane)
        qq_png = os.path.join(graphs, "diabetes_qq.png")
        hist_png = os.path.join(graphs, "diabetes_histogram_rezidui.png")
        pred_png = os.path.join(graphs, "diabetes_rezidua_vs_predikce.png")
        bmi_png = os.path.join(graphs, "diabetes_rezidua_vs_bmi.png")
        analyzer.qq_residuals(save_path=qq_png)
        analyzer.histogram_residuals(save_path=hist_png)
        analyzer.residuals_vs_prediction(save_path=pred_png)
        analyzer.residuals_vs_x(bmi, save_path=bmi_png, x_label="bmi (standardizovano)")
        print(f"  grafy ulozeny: {qq_png}, {hist_png}, {pred_png}, {bmi_png}")
        vysledek_testu = analyzer.residual_normality(alpha=alpha)
        print(f"  Shapiruv-Wilkuv test: W = {vysledek_testu['statistic']:.4f}, "
              f"p = {vysledek_testu['p_value']:.4f}")
        if vysledek_testu["normal"]:
            print(f"  Zaver: normalita rezidui nezamitnuta (p > {alpha}).")
        else:
            print(f"  Zaver: normalita rezidui zamitnuta (p <= {alpha}); "
                  "posudte ji spolu s QQ grafem a histogramem.")
    except NotImplementedError as exc:
        faze_neni_hotova(exc)
    except Exception as exc:
        chyba_implementace(exc)


def faze_a6_ulozeni(
    cfg: ExperimentConfig, diabetes: DiabetesData | None,
    vysledek: DiabetesModel | None, duvod: str,
    ) -> None:
    """Faze A6 -- ulozeni state_dict do models/ a nacteni do NOVE instance site."""
    banner("Faze A6: Ulozeni modelu a kontrolni nacteni")
    if vysledek is None or diabetes is None:
        preskoceno(duvod)
        return
    tolerance = cfg.diagnostics.load_tolerance
    cesta = os.path.join(cfg.paths.models_dir, "mlp_diabetes.pt")
    os.makedirs(cfg.paths.models_dir, exist_ok=True)
    try:
        vysledek.trainer.save(cesta)
        print(f"  state_dict ulozen: {cesta}")

        # Nova instance STEJNE architektury; vahy se do ni nahraji ze souboru.
        nova_sit = MLP(diabetes.x_train.shape[1], cfg.regression.hidden_dims)
        nacteny = Trainer.load(cesta, nova_sit, cfg.regression.training)
        x_test = vstupni_matice(TabularDataset(diabetes.x_test, diabetes.y_test))
        rozdil = float((nacteny.predict(x_test) - vysledek.trainer.predict(x_test)).abs().max())
        if rozdil < tolerance:
            print(f"  [OK] Nactena sit dava stejne predikce (max. rozdil {rozdil:.1e}).")
        else:
            print(f"  [CHYBA] Nactena sit se lisi (max. rozdil {rozdil:.1e}, "
                  f"diagnostics.load_tolerance = {tolerance:.0e}).")
    except NotImplementedError as exc:
        faze_neni_hotova(exc)
    except pickle.UnpicklingError as exc:
        chyba_implementace(exc)
        print("  -> Ulozte jen state_dict() modelu (torch.save(self.model.state_dict(), path)), "
              "ne cely model.")
    except Exception as exc:
        chyba_implementace(exc)


# ============================================================================ #
#  Cast B: predpoved casove rady                                               #
# ============================================================================ #
def faze_b1_data(cfg: ExperimentConfig) -> SeriesData | None:
    """Faze B1 -- teploty: nacteni, chronologicke rozdeleni a standardizace.

    Standardizace pouziva prumer a odchylku JEN z trenovaci casti (jinak by
    do uceni unikla informace z budoucnosti).

    Parametry
    ---------
    cfg : ExperimentConfig
        Konfigurace experimentu.

    Navratova hodnota
    -----------------
    SeriesData | None
        Rada a jeji casti, nebo ``None`` (chybejici ci poskozeny soubor,
        okno delsi nez cast rady); cela cast B se pak preskoci.
    """
    banner("Faze B1: Data -- denni minimalni teploty (chronologicke rozdeleni)")
    w = cfg.forecast.window_size
    try:
        serie = load_temperature_series(cfg.paths.temperature_csv)
        train, test = chronological_split(serie, train_frac=cfg.forecast.train_frac)
    except Exception as exc:
        chyba_dat(exc)
        print(f"  -> Cast B se preskoci (potrebuje platny soubor "
              f"{cfg.paths.temperature_csv}, klic paths.temperature_csv); "
              "cast A i zaverecny banner probehnou.")
        return None

    if w >= len(train) or w >= len(test):
        print(f"  [CHYBA KONFIGURACE] forecast.window_size = {w} musi byt mensi nez delka "
              f"trenovaci casti ({len(train)}) i testovaci casti ({len(test)}).")
        print("  -> Cast B se preskoci; snizte forecast.window_size v config.yaml.")
        return None

    mean, std = float(train.mean()), float(train.std())
    print(f"  rada: {len(serie)} dnu, rozsah {serie.min():.1f} az {serie.max():.1f} stupnu C")
    print(f"  trenovaci cast: {len(train)} dnu (prvnich {100 * cfg.forecast.train_frac:.0f} %), "
          f"rozsah {train.min():.1f} az {train.max():.1f}")
    print(f"  testovaci cast: {len(test)} dnu (posledni usek rady), "
          f"rozsah {test.min():.1f} az {test.max():.1f}")
    print(f"  standardizace z trenovaci casti: prumer {mean:.2f}, odchylka {std:.2f}")
    return SeriesData(train, test, (train - mean) / std, (test - mean) / std, mean, std)


def faze_b2_uceni(
    cfg: ExperimentConfig, serie: SeriesData | None,
    ) -> tuple[ForecastModel | None, str]:
    """Faze B2 -- okenkovani rady a uceni site; ulozeni modelu do models/.

    Okna se michaji jen uvnitr trenovaciho loaderu (poradi oken v davkach).
    To je v poradku: kazde okno drzi hodnoty ve spravnem poradi a vsechna
    trenovaci okna lezi PRED delicim bodem, takze budoucnost do uceni
    neprosakuje. Zakazane je michat pred rozdelenim rady.

    Parametry
    ---------
    cfg : ExperimentConfig
        Konfigurace experimentu.
    serie : SeriesData | None
        Data z faze B1.

    Navratova hodnota
    -----------------
    tuple[ForecastModel | None, str]
        Naucena sit a testovaci okna (nebo ``None``) a duvod pro zavisle faze.
    """
    banner("Faze B2: Okenkovani rady a uceni site")
    if serie is None:
        duvod = "Rada teplot nebyla nactena (faze B1 neprobehla)."
        preskoceno(duvod)
        return None, duvod
    duvod = "Sit nebyla naucena (faze B2 neprobehla)."
    fc = cfg.forecast
    print(f"  okno {fc.window_size} dni -> dalsi den; architektura "
          f"{fc.window_size}-{'-'.join(str(h) for h in fc.hidden_dims)}-1; "
          f"lr {fc.training.lr}, epoch {fc.training.epochs}")
    try:
        train_ds = WindowedDataset(serie.train_std, fc.window_size)
        test_ds = WindowedDataset(serie.test_std, fc.window_size)
        print(f"  okna: trenovaci {len(train_ds)}, testovaci {len(test_ds)}")
        torch.manual_seed(cfg.seed)
        model = MLP(fc.window_size, fc.hidden_dims)
        trainer, history = nauc_model(
            model, train_ds, test_ds, fc.training, cfg.paths.logs_dir, cfg.diagnostics,
            section="forecast",
        )
        curve_png = os.path.join(cfg.paths.graphs_dir, "forecast_krivka_uceni.png")
        plot_loss_curve(history, save_path=curve_png,
                        title="Predpoved teplot: krivka uceni (standardizovane teploty)")
        print(f"  graf ulozen: {curve_png}")
        vysledek = ForecastModel(trainer, test_ds)
    except NotImplementedError as exc:
        faze_neni_hotova(exc)
        return None, duvod
    except UceniDivergovalo as exc:
        print(f"  [CHYBA UCENI] {exc} (forecast.lr)")
        return None, "Uceni v fazi B2 divergovalo -- zmensete forecast.lr v config.yaml."
    except Exception as exc:
        chyba_implementace(exc)
        return None, duvod

    # Ulozeni zvlast: nedokoncene save nesmi zahodit hotove uceni.
    cesta = os.path.join(cfg.paths.models_dir, "mlp_forecast.pt")
    os.makedirs(cfg.paths.models_dir, exist_ok=True)
    try:
        trainer.save(cesta)
        print(f"  state_dict ulozen: {cesta}")
    except NotImplementedError as exc:
        faze_neni_hotova(exc)
    except Exception as exc:
        chyba_implementace(exc)
    return vysledek, duvod


def _predpoved_ve_stupnich(serie: SeriesData, vysledek: ForecastModel) -> np.ndarray:
    """Predpovedi na testovacich oknech prevedene zpet na stupne C.

    Parametry
    ---------
    serie : SeriesData
        Rada se statistikami standardizace.
    vysledek : ForecastModel
        Naucena sit a testovaci okna.

    Navratova hodnota
    -----------------
    np.ndarray
        Predpovedi tvaru ``(len(test) - window_size,)`` v puvodnich jednotkach.
    """
    return predikuj(vysledek.trainer, vysledek.test_ds) * serie.std + serie.mean


def faze_b3_baseline(
    cfg: ExperimentConfig, serie: SeriesData | None,
    vysledek: ForecastModel | None, duvod: str,
    ) -> None:
    """Faze B3 -- MAE a RMSE modelu proti persistence baseline na testovaci casti."""
    banner("Faze B3: Model proti persistence baseline (zitra jako dnes)")
    if serie is None or vysledek is None:
        preskoceno(duvod)
        return
    w = cfg.forecast.window_size
    try:
        predpoved = _predpoved_ve_stupnich(serie, vysledek)
        cile = serie.test[w:]                          # cile oken = test[window_size:]
        baseline = persistence_baseline(serie.test, w)
        if baseline.shape != cile.shape:
            print(f"  [CHYBA IMPLEMENTACE] persistence_baseline vratila tvar {baseline.shape}, "
                  f"ocekavan {cile.shape} (jedna predpoved na kazdy cil okna).")
            print("  -> Zkontrolujte rezani rady (README, odd. 7); pipeline pokracuje dal.")
            return
        mae_model, mae_base = mae(cile, predpoved), mae(cile, baseline)
        rmse_model, rmse_base = mse(cile, predpoved) ** 0.5, mse(cile, baseline) ** 0.5

        print(f"  testovacich predpovedi: {len(cile)} (jeden krok dopredu)")
        print(f"  {'':22s}{'MAE [C]':>10s}{'RMSE [C]':>11s}")
        print(f"  {'sit (MLP)':22s}{mae_model:10.3f}{rmse_model:11.3f}")
        print(f"  {'persistence baseline':22s}{mae_base:10.3f}{rmse_base:11.3f}")
        if mae_base <= cfg.diagnostics.zero_error_tolerance:
            print("  [POZOR] persistence baseline ma nulovou chybu -- vraci samotne cile, "
                  "ne hodnotu z predchoziho dne (chyba o jedna).")
            return
        rozdil = 100.0 * (mae_base - mae_model) / mae_base
        if rozdil >= 0:
            print(f"  Sit ma MAE o {rozdil:.1f} % nizsi nez baseline.")
        else:
            print(f"  Sit ma MAE o {-rozdil:.1f} % vyssi nez baseline.")
        if mae_model < mae_base:
            print("  [OK] Verdikt: sit porazila persistence baseline.")
        else:
            print("  [POZOR] Verdikt: sit persistence baseline NEPORAZILA -- "
                  "bez modelu bychom predpovedeli stejne dobre nebo lepe.")
    except NotImplementedError as exc:
        faze_neni_hotova(exc)
    except Exception as exc:
        chyba_implementace(exc)


def faze_b4_grafy(
    cfg: ExperimentConfig, serie: SeriesData | None,
    vysledek: ForecastModel | None, duvod: str,
    ) -> None:
    """Faze B4 -- graf predpovedi v case a graf rezidui v case."""
    banner("Faze B4: Grafy predpovedi a rezidui v case")
    if serie is None or vysledek is None:
        preskoceno(duvod)
        return
    try:
        predpoved = _predpoved_ve_stupnich(serie, vysledek)
        cile = serie.test[cfg.forecast.window_size:]
        forecast_png = os.path.join(cfg.paths.graphs_dir, "forecast_predpoved.png")
        resid_png = os.path.join(cfg.paths.graphs_dir, "forecast_rezidua_v_case.png")
        plot_forecast(cile, predpoved, save_path=forecast_png)
        plot_residuals_in_time(cile - predpoved, save_path=resid_png)
        print(f"  grafy ulozeny: {forecast_png}, {resid_png}")
    except NotImplementedError as exc:
        faze_neni_hotova(exc)
    except Exception as exc:
        chyba_implementace(exc)


def main() -> None:
    """Spusti celou pipeline cviceni 11 s ochrannymi bloky u kazde faze."""
    banner("CVICENI 11 -- Regrese a predpoved neuronovou siti (PyTorch) -- start")

    # --- Config guard -----------------------------------------------------------
    try:
        cfg = load_config()
    except (ValueError, FileNotFoundError) as exc:
        print(f"[CHYBA KONFIGURACE] {exc}")
        sys.exit(1)

    # Cast A: regrese
    data_a = faze_a1_data(cfg)
    sinus, diabetes = data_a if data_a is not None else (None, None)
    faze_a2_sinusovka(cfg, sinus)
    vysledek_a, duvod_a = faze_a3_diabetes_uceni(cfg, diabetes)
    faze_a4_diabetes_metriky(cfg, diabetes, vysledek_a, duvod_a)
    faze_a5_diabetes_rezidua(cfg, diabetes, vysledek_a, duvod_a)
    faze_a6_ulozeni(cfg, diabetes, vysledek_a, duvod_a)

    # Cast B: predpoved casove rady
    serie = faze_b1_data(cfg)
    vysledek_b, duvod_b = faze_b2_uceni(cfg, serie)
    faze_b3_baseline(cfg, serie, vysledek_b, duvod_b)
    faze_b4_grafy(cfg, serie, vysledek_b, duvod_b)

    banner("CVICENI 11 -- konec")


if __name__ == "__main__":
    main()
