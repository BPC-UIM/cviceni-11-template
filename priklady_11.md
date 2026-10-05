# Papírové příklady — Cvičení 11

Regrese a předpověď časové řady neuronovou sítí: regresní metriky, past
tvarů, okénkování řady, persistence baseline, únik dat při míchání a jeden
krok gradientního sestupu v PyTorchi. Příklady procvičují **přesně tu notaci
a ty výpočty**, se kterými pracujete v `src/metrics.py`, `src/datasets.py`,
`src/model.py` a `src/trainer.py`. Čísla jsou volena tak, aby se dala
spočítat na papíře.

**Řešení nejsou součástí repozitáře.** Výsledky si ověřte u vyučujícího nebo
výpočtem v `numpy` (nejlépe přímo funkcemi a třídami ze `src/`, jakmile je
doplníte, viz konec souboru).

Značení je stejné jako v kódu:

- `y_true`, `y_pred`: skutečné hodnoty a předpovědi, pole tvaru `(n,)`;
  $y_i$ a $\hat{y}_i$ jsou jejich prvky, $\bar{y}$ průměr **skutečných**
  hodnot. Chyba (reziduum) je $e_i = y_i - \hat{y}_i$.
- `mse` $= \frac{1}{n}\sum_i e_i^2$, RMSE $= \sqrt{\text{MSE}}$,
  `mae` $= \frac{1}{n}\sum_i |e_i|$.
- `r2` $= 1 - SS_{\text{res}}/SS_{\text{tot}}$, kde
  $SS_{\text{res}} = \sum_i (y_i - \hat{y}_i)^2$ a
  $SS_{\text{tot}} = \sum_i (y_i - \bar{y})^2$. $R^2 = 0$ odpovídá předpovědi
  průměrem $\bar{y}$.
- Dávka z `DataLoader`: `x_batch` tvaru `(B, n_features)`, `y_batch` tvaru
  `(B, 1)`; výstup modelu `y_pred` má tvar `(B, 1)`.
- `WindowedDataset(series, window_size)`: vzorek `i` má vstup
  `x = series[i : i + window_size]` (tvar `(window_size,)`) a cíl
  `y = series[i + window_size]` (tvar `(1,)`); oken je
  `len(series) - window_size`. Indexy počítáme od 0 jako v Pythonu.
- `persistence_baseline(series, window_size)`: naivní předpověď „zítra bude
  jako dnes", pole tvaru `(len(series) - window_size,)`, jehož prvek `i`
  patří cíli okna `i`.
- `MLP(input_dim, hidden_dims)`: `nn.Linear` + `nn.Tanh` za každou skrytou
  vrstvou, poslední `nn.Linear` bez aktivace. Architekturu zapisujeme
  zkráceně, např. 1-32-16-1.
- Krok SGD: $w \leftarrow w - \eta\,\partial L/\partial w$, ztráta je MSE
  (průměr přes dávku).

---

## Příklad 1 — MSE, MAE, RMSE a $R^2$ ručně

Skutečné hodnoty jsou $\mathbf{y} = (2,\ 4,\ 6,\ 8)$. Čtyři modely
předpověděly:

| model | $\hat{\mathbf{y}}$ |
|:---:|:---|
| A | $(3,\ 4,\ 5,\ 8)$ |
| B | $(5,\ 5,\ 5,\ 5)$ |
| C | $(8,\ 6,\ 4,\ 2)$ |
| D | $(3,\ 4,\ 5,\ 12)$ |

**Úkoly:**

- **a)** Spočítejte $\bar{y}$ a $SS_{\text{tot}}$.
- **b)** Pro každý model vypište rezidua $e_i$ a spočítejte MSE, RMSE, MAE,
  $SS_{\text{res}}$ a $R^2$. Výsledky zapište do tabulky.
- **c)** Proč vyšlo pro model B právě $R^2 = 0$? Co model B „umí"?
- **d)** Model C má $R^2 < 0$. Co to znamená slovně? Je pravda, že $R^2$ leží
  vždy mezi 0 a 1?
- **e)** Model E předpovídá konstantu $\hat{y}_i = 6$. Spočítejte jeho $R^2$.
  Dokažte, že ze všech konstantních předpovědí $\hat{y}_i = c$ má největší
  $R^2$ právě $c = \bar{y}$ (nápověda: rozepište $\sum_i (y_i - c)^2$ pomocí
  $y_i - \bar{y}$ a $\bar{y} - c$). Kolik je to maximum?
- **f)** Model D se od A liší jedinou předpovědí. Kolikrát vzrostla MSE,
  kolikrát MAE a kolikrát RMSE? Která metrika je na jednu odlehlou chybu
  nejcitlivější a proč?
- **g)** Proč `r2` počítá $SS_{\text{tot}}$ z průměru **skutečných** hodnot
  `y_true`, a ne z průměru předpovědí? Spočítejte „$R^2$" s průměrem
  předpovědí pro model E z bodu e). Co by tato chybná varianta vracela pro
  **každou** konstantní předpověď?

---

## Příklad 2 — Past tvarů: `(n,)` minus `(n, 1)`

Student má skutečné hodnoty `y_true = np.array([1.0, 2.0, 3.0])` (tvar
`(3,)`) a předpovědi modelu jako sloupec
`y_pred = np.array([[1.0], [2.0], [3.0]])` (tvar `(3, 1)`). Předpovědi jsou
tedy **dokonalé**.

**Úkoly:**

- **a)** Podle pravidel broadcastingu určete tvar výsledku
  `y_true - y_pred` a vypište celou matici. Který index matice patří
  `y_true` a který `y_pred`?
- **b)** Spočítejte `np.mean((y_true - y_pred) ** 2)` a
  `np.mean(np.abs(y_true - y_pred))`. Kolik je správná MSE a MAE?
- **c)** Totéž nastane v PyTorchi: `nn.MSELoss()(y_pred, y_batch)`
  s `y_pred` tvaru `(B, 1)` a `y_batch` tvaru `(B,)`. Kolik dvojic se
  zprůměruje pro `B = 32`? Proč program nespadne a co jediné uživatele
  varuje?
- **d)** Ukažte, že taková „ztráta" je pro každou předpověď $\hat{y}_i$
  minimální, když $\hat{y}_i$ je rovno **průměru celé dávky** cílů. Co se
  tedy síť s chybnými tvary naučí? Ověřte na dávce cílů $(1,\ 2,\ 6)$.
- **e)** Které místo v `Trainer._train_step` a v metrikách tuto chybu
  zachytí? Proč je lepší program zastavit assertem než spoléhat na
  varování?

---

## Příklad 3 — Okénkování řady ručně

Denní řada (např. teploty ve °C) má deset hodnot:

$$\texttt{series} = (12,\ 14,\ 13,\ 15,\ 17,\ 16,\ 18,\ 20,\ 19,\ 21),\qquad \texttt{window\_size} = 3.$$

**Úkoly:**

- **a)** Kolik oken vytvoří `WindowedDataset(series, 3)`? Vypište všechny
  dvojice `(x, y)` a tvary `x` a `y` jednoho vzorku.
- **b)** `DataLoader(ds, batch_size=3, shuffle=False)`: kolik dávek
  vznikne a jaké mají tvary `x_batch` a `y_batch`? Proč má poslední dávka
  jiný tvar?
- **c)** Kolik oken vznikne pro `window_size` $= 1,\ 5,\ 9$ a co se stane
  pro `window_size = 10`?
- **d)** Řada se nejprve rozdělí funkcí `chronological_split(series,
  train_frac=0.6)` a teprve pak se každá část okénkuje zvlášť (tak to dělá
  pipeline). Vypište trénovací a testovací část a jejich okna. Kolik oken
  oproti okénkování celé řady „zmizelo" a proč? Které hodnoty testovací
  části nikdy nejsou cílem?
- **e)** *(stará chyba)* Starší verze cvičení okénkovala **časovou osu**:
  vstupem okna `i` byly indexy dnů `(i, i+1, i+2)`, cílem hodnota
  `series[i + 3]`. Vypište takto vzniklé dvojice pro celou řadu. Jakou
  informaci síť ve vstupu dostává a jakou ne? Kolik „různých čísel" nese
  vstup jednoho okna?
- **f)** Při chronologickém rozdělení z bodu d) by chybná síť z bodu e) na
  testovacím okně dostala indexy, které v trénování nikdy neviděla. Které
  to jsou? Proč takovou síť nelze použít k předpovědi budoucnosti, i kdyby
  na trénovacích datech chybovala málo?

---

## Příklad 4 — Persistence baseline ručně

Testovací úsek řady a velikost okna:

$$\texttt{series} = (20,\ 21,\ 23,\ 22,\ 24,\ 25,\ 27,\ 26),\qquad \texttt{window\_size} = 3.$$

**Úkoly:**

- **a)** Vypište cíle všech oken (`y` z `WindowedDataset(series, 3)`) a ke
  každému cíli naivní předpověď „zítra jako dnes". Kolik předpovědí je?
- **b)** Vyjádřete prvek `baseline[i]` jako prvek řady `series[?]`
  (index jako funkci `i` a `window_size`). Ověřte, že je to poslední hodnota
  okna `i`.
- **c)** Spočítejte MAE a RMSE baseline.
- **d)** Dva modely předpověděly pro tytéž cíle:
  $\hat{\mathbf{y}}^{(1)} = (21,\ 25,\ 24,\ 28,\ 25)$ a
  $\hat{\mathbf{y}}^{(2)} = (22,\ 23,\ 25,\ 28,\ 30)$. Spočítejte jejich MAE
  a RMSE. Porazil baseline model 1? Porazil ji model 2? Spočítejte, o kolik
  procent má každý model nižší MAE než baseline (vzorec pipeline:
  $100\cdot(\text{MAE}_{\text{base}} - \text{MAE}_{\text{model}})/\text{MAE}_{\text{base}}$).
  Co říká rozpor mezi MAE a RMSE u modelu 2?
- **e)** *(chyba o jedna, varianta A)* Student omylem vrátí jako „baseline"
  samotné cíle. Jaké vyjde MAE baseline a co pak pipeline řekne o každém
  modelu?
- **f)** *(chyba o jedna, varianta B)* Student vrátí hodnoty o dva dny
  starší („zítra jako předevčírem"). Vypište je a spočítejte MAE a RMSE.
  O kolik procent by pak model 2 „porazil" baseline? Proč je tahle chyba
  zrádnější než varianta A?
- **g)** V pipeline má testovací část 730 dní a `window_size = 14`. Kolik
  testovacích předpovědí (cílů) se porovnává?

---

## Příklad 5 — Proč se časová řada nesmí míchat

Řada má deset hodnot $s_0, s_1, \dots, s_9$ a `window_size = 3`. Okno
$W_i$ má vstup $(s_i, s_{i+1}, s_{i+2})$ a cíl $s_{i+3}$.

**Úkoly:**

- **a)** Kolik oken vznikne? V kolika oknech je hodnota $s_5$ **vstupem**
  a v kolika **cílem**?
- **b)** Okna se náhodně rozdělí: testovací jsou $W_1$ a $W_4$, ostatní
  trénovací. Pro každé testovací okno najděte trénovací okna, jejichž
  **vstup obsahuje cíl** testovacího okna. Kolik vstupních hodnot sdílí
  $W_4$ s trénovacími okny $W_3$ a $W_5$?
- **c)** Vysvětlete na bodu b), proč náhodné rozdělení oken vede
  k optimisticky nízké testovací chybě. Jakou informaci by model při
  skutečné předpovědi budoucnosti neměl?
- **d)** Nyní rozdělení chronologické: trénovací $W_0, \dots, W_4$,
  testovací $W_5, W_6$. Je některý testovací cíl vstupem trénovacího okna?
  Testovací okno $W_5$ přitom obsahuje hodnoty $s_5, s_6$, které byly
  trénovacími cíli. Je to únik? Zdůvodněte.
- **e)** Pipeline volá `DataLoader(train_ds, shuffle=True)`. Proč je
  míchání **trénovacích oken v dávkách** v pořádku, zatímco míchání **před
  rozdělením** ne?
- **f)** Spojte s Cvičením 06: co tam „unikalo" do učení a co uniká zde?

---

## Příklad 6 — PyTorch: počet parametrů a jeden krok SGD ručně

**Úkoly (parametry):**

- **a)** Vrstva `nn.Linear(n_in, n_out)` má matici vah tvaru
  `(n_out, n_in)` a bias tvaru `(n_out,)`. Odvoďte obecný vzorec pro počet
  parametrů MLP s architekturou $n_0$-$n_1$-…-$n_L$.
- **b)** Spočítejte počet parametrů pro 1-32-16-1 (sinusovka),
  10-32-16-1 (diabetes), 14-32-16-1 (předpověď teplot), 1-16-1 a 14-1
  (lineární model bez skryté vrstvy). Kolik parametrů měla stará síť
  1-70-70-70-70-70-70-1 (šest skrytých vrstev po 70)?

**Úkoly (krok učení):** Model má jedinou váhu bez biasu, $\hat{y} = w\,x$.
Dávka obsahuje dva vzorky: $x = (1,\ 2)$, $y = (3,\ 6)$. Počáteční váha
$w_0 = 1$, krok učení $\eta = 0{,}05$, ztráta MSE
$L(w) = \frac{1}{2}\sum_{i=1}^{2} (w x_i - y_i)^2$.

- **c)** Odvoďte $\partial L/\partial w$ obecně. Pro jakou váhu je
  gradient nulový?
- **d)** Proveďte první krok: $\hat{\mathbf{y}}$, $L$, $\partial L/\partial w$
  a $w_1$. Jakou hodnotu vrátí `_train_step` (ztráta před, nebo po
  aktualizaci)?
- **e)** Proveďte druhý a třetí krok **se** `zero_grad` (do `.grad` se
  pokaždé zapíše čerstvý gradient). Ukažte, že odchylka $w - 3$ se
  v každém kroku násobí stejným číslem. Kterým? Po kolika krocích poprvé
  platí $|w - 3| < 0{,}01$?
- **f)** Proveďte druhý a třetí krok **bez** `zero_grad`: PyTorch nový
  gradient přičte k hodnotě v `.grad`. Kolik je `.grad` ve druhém
  a třetím kroku a jaké jsou váhy $w_2$, $w_3$? Porovnejte s bodem e).
- **g)** *(s kalkulačkou)* Pokračujte bez `zero_grad` ještě několik kroků.
  Konverguje váha k 3? Vysvětlete, proč nahromaděný gradient nezmizí ani
  v okamžiku, kdy $w = 3$.

---

## Příklad 7 — Standardizace cíle a převod metrik zpět *(volitelný)*

Trénovací cíle jsou $(100,\ 100,\ 200,\ 200)$. Síť se učí na standardizovaném
cíli $z = (y - \bar{y}_{\text{train}})/s_y$, kde $s_y$ je směrodatná
odchylka trénovacích cílů (`np.std`, tedy dělení $n$). Testovací cíle jsou
$(100,\ 150,\ 200,\ 250)$ a síť pro ně předpověděla standardizované hodnoty
$\hat{z} = (-0{,}8;\ 0{,}2;\ 0{,}6;\ 2{,}0)$.

**Úkoly:**

- **a)** Spočítejte `y_mean` a `y_std` z trénovacích cílů a standardizujte
  testovací cíle.
- **b)** Spočítejte MSE, MAE a $R^2$ ve standardizovaných jednotkách.
- **c)** Převeďte předpovědi zpět do původních jednotek
  ($\hat{y} = \hat{z}\cdot s_y + \bar{y}_{\text{train}}$) a spočítejte MSE,
  MAE, RMSE a $R^2$ znovu.
- **d)** Jakým faktorem se změnily MAE, RMSE a MSE? Proč se $R^2$ nezměnilo?
- **e)** Student převede MAE zpět vzorcem `mae_std * y_std + y_mean`. Co
  vyjde a proč je to nesmysl?

---

## Příklady k procvičení

Následující úlohy procvičují **tentýž postup jako příklady 1, 3, 4 a 6**,
jen na jiných datech.

**Cvičení P1 — metriky.** $\mathbf{y} = (1,\ 3,\ 5,\ 7,\ 9)$,
$\hat{\mathbf{y}} = (2,\ 3,\ 4,\ 8,\ 8)$. Spočítejte MSE, RMSE, MAE a $R^2$.
Pak totéž pro konstantní předpověď $\hat{y}_i = 4$.
*Na co se zaměřit:* proč je $R^2$ konstantní předpovědi záporné, ačkoli její
MAE není „katastrofální"?

**Cvičení P2 — okna a baseline.** $\texttt{series} = (5,\ 7,\ 6,\ 8,\ 9,\ 7,\ 10)$,
`window_size = 2`. Vypište všechna okna `(x, y)`, cíle a persistence
baseline, spočítejte MAE a RMSE baseline. Porazí ji model
$\hat{\mathbf{y}} = (7,\ 7,\ 8,\ 8,\ 9)$?
*Na co se zaměřit:* jaké $R^2$ má na této řadě baseline? Co to říká
o rozkolísané řadě bez trendu?

**Cvičení P3 — lineární trend.** $\texttt{series} = (0,\ 2,\ 4,\ 6,\ 8,\ 10)$,
`window_size = 2`. Spočítejte MAE persistence baseline a MAE předpovědi
„zítra = dnes + (dnes − včera)".
*Na co se zaměřit:* proč persistence na řadě s trendem systematicky
zaostává a jaké znaménko mají její rezidua?

**Cvičení P4 — parametry.** Spočítejte počet parametrů sítí 5-8-1, 3-4-4-2
a 7-1. Která z nich je lineární model a kolik vah a biasů má?

> **Řešení nejsou k dispozici.** Výsledky si ověřte na zprovozněném
> repozitáři, jakmile doplníte metriky, `WindowedDataset`
> a `persistence_baseline`. Například pro příklady 1, 3 a 4:
>
> ```python
> import numpy as np
> from src import WindowedDataset, mae, mse, persistence_baseline, r2
>
> y = np.array([2.0, 4.0, 6.0, 8.0])
> for jmeno, y_pred in [("A", [3, 4, 5, 8]), ("B", [5, 5, 5, 5])]:
>     y_pred = np.array(y_pred, dtype=float)
>     print(jmeno, mse(y, y_pred), mse(y, y_pred) ** 0.5, mae(y, y_pred), r2(y, y_pred))
>
> ds = WindowedDataset(np.array([12, 14, 13, 15, 17, 16, 18, 20, 19, 21.0]), 3)
> print(len(ds), [(x.tolist(), y.tolist()) for x, y in (ds[i] for i in range(len(ds)))])
>
> rada = np.array([20, 21, 23, 22, 24, 25, 27, 26.0])
> cile, baseline = rada[3:], persistence_baseline(rada, 3)
> print(cile, baseline, mae(cile, baseline), mse(cile, baseline) ** 0.5)
> ```
