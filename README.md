# Cvičení 11: Regrese a predikce neuronovou sítí — PyTorch

Jedenácté praktické cvičení předmětu **Umělá inteligence v medicíně** uzavírá
blok neuronových sítí. V Cvičení 08 jsme sestavili neuron, v Cvičení 09 síť
s ručně navrženými vahami a v Cvičení 10 jsme síť naučili gradientním
sestupem, přičemž zpětné šíření chyby jsme napsali ručně v NumPy. Nyní
přecházíme na knihovnu **PyTorch**. Gradienty za nás počítá automatické
derivování (*autograd*) a ruční zpětný průchod z Cvičení 10 nahradí několik
řádků kódu.

Síť přitom poprvé nepoužijeme ke klasifikaci, ale ke dvěma úlohám se
**spojitým výstupem**:

- **Část A, regrese.** Proložení zašuměné sinusovky (jeden příznak, výsledek
  je vidět) a odhad progrese diabetu z deseti klinických příznaků.
- **Část B, předpověď časové řady.** Z posledních dnů řady denních minimálních
  teplot předpovídáme následující den.

Z Cvičení 10 se **nekopíruje žádný kód**. Návaznost je koncepční: každá
součást, kterou jste psali ručně, má v PyTorchi přímý protějšek (tabulka
v odd. 1 teorie). Uvolněnou kapacitu věnujeme tomu, co je na obou úlohách
nové: správnému hodnocení regrese, okénkování časové řady a poctivému
srovnání s naivní předpovědí.

---

## Obsah

1. [Cíle cvičení](#cíle-cvičení)
2. [Struktura repozitáře](#struktura-repozitáře)
3. [Instalace a spuštění](#instalace-a-spuštění)
4. [Teoretický základ](#teoretický-základ)
5. [Konfigurace projektu](#konfigurace-projektu)
6. [Pokyny k vypracování](#pokyny-k-vypracování)
7. [Lokální testování](#lokální-testování)
8. [Doplňkové (papírové) příklady](#doplňkové-papírové-příklady)
9. [Odevzdání](#odevzdání)

---

## Cíle cvičení

Po dokončení tohoto cvičení student:

1. **Převede ruční síť do PyTorche.** Ke každé součásti z Cvičení 08–10
   (lineární vrstva, aktivace, zpětný průchod, krok gradientního sestupu,
   dávkování) zná její protějšek v PyTorchi.
2. **Sestaví model jako `nn.Module`.** Z `nn.Linear` a `nn.Tanh` složí
   `nn.Sequential` a napíše metodu `forward`.
3. **Napíše tréninkový krok.** Zná pořadí `zero_grad` → dopředný průchod →
   ztráta → `backward` → `step` a umí vysvětlit, co každý řádek dělá.
4. **Rozlišuje regresi a klasifikaci.** Ví, proč má regresní síť **lineární
   výstupní vrstvu** a proč se učí střední kvadratickou chybou.
5. **Hodnotí regresi.** Z definice spočítá MSE, MAE a koeficient determinace
   $R^2$ a umí je interpretovat.
6. **Čte reziduální diagnostiku.** Z grafů reziduí pozná, zda model data
   skutečně vystihuje, nebo má jen přijatelné $R^2$.
7. **Okénkuje časovou řadu nad signálem.** Z řady vytvoří dvojice
   (posledních $k$ hodnot, následující hodnota) pomocí vlastního `Dataset`.
8. **Dělí časovou řadu chronologicky.** Umí vysvětlit, proč je náhodné
   zamíchání časové řady únikem dat.
9. **Srovnává model s naivní předpovědí.** Ví, že předpovědní model má cenu
   jen tehdy, když porazí *persistence baseline* („zítra bude jako dnes").
10. **Ukládá model jako `state_dict`.** Rozumí tomu, proč se ukládají naučené
    parametry, a ne celý objekt.

---

## Struktura repozitáře

```
cviceni-11-template/
├── cviceni_11.py              # Pipeline, celá PŘEDVYPLNĚNA: jen fáze A1–B4 a main()
├── config.yaml                # VŠE nastavitelné: cesty, data, sítě, učení, prahy kontrol
├── priklady_11.md             # Papírové příklady, BEZ řešení
├── requirements.txt           # Python závislosti (zamčené verze, včetně torch)
├── .gitignore
├── README.md
├── src/
│   ├── __init__.py            # Re-exporty balíčku (neupravujte)
│   ├── model.py               # ÚKOL: MLP(nn.Module), __init__ a forward
│   ├── trainer.py             # ÚKOL: _train_step, save, load; smyčka učení a logování předvyplněny
│   ├── datasets.py            # VZOR: TabularDataset; ÚKOL: WindowedDataset
│   ├── metrics.py             # ÚKOL: mse, mae, r2, persistence_baseline (NumPy, od nuly)
│   └── diagnostics.py         # ResidualAnalyzer (PŘEDVYPLNĚNO)
├── dataio/
│   ├── __init__.py            # Re-exporty balíčku (neupravujte)
│   ├── loaders.py             # Diabetes, sinusovka, teploty, rozdělení dat (předvyplněno)
│   ├── config_manager.py      # Dataclassy + load_config + validate_config (předvyplněno)
│   └── plotting.py            # Proložení, křivka učení, předpověď v čase (předvyplněno)
├── utils/                     # Pomocný kód pipeline (celý PŘEDVYPLNĚNÝ, neupravujte)
│   ├── __init__.py            # Re-exporty balíčku
│   ├── reporting.py           # Hlášky [NENI HOTOVO], [CHYBA ...], [PRESKOCENO], bannery
│   ├── containers.py          # Přepravky výsledků, které si fáze předávají
│   └── training.py            # Učení s kontrolou divergence a přeučení, predikce
├── data/
│   └── daily_min_temperatures.csv   # Denní minimální teploty, Melbourne 1981–1990
├── graphs/                    # Výstupní grafy (generují se automaticky)
│   └── .gitkeep
├── logs/                      # Log učení trainer.log (generuje se automaticky)
│   └── .gitkeep
├── models/                    # Naučené modely, state_dict v souborech .pt
│   └── .gitkeep
└── test_cviceni_11.py         # Automatické testy (pytest)
```

> **Poznámka k souborům `__init__.py`:** Každá složka s Python kódem (`src/`,
> `dataio/`, `utils/`) obsahuje `__init__.py`, který ji označuje jako balíček a definuje
> veřejné API. Díky tomu lze psát `from src import MLP` místo
> `from src.model import MLP`. **Tyto soubory neupravujte.**

> **Žádná brána z minulého cvičení.** Do tohoto repozitáře se nevkládá žádné
> řešení z Cvičení 10. Ruční třídy `Linear`, `Neuron` a `Sequential` by se
> v PyTorchi nepoužily. Všechny hlášky nedokončených částí proto začínají
> `Úkol:` a týkají se jen tohoto cvičení.

> **Jeden model, jeden `Trainer`, dvě úlohy.** Třída `MLP` i třída `Trainer`
> jsou společné pro regresi i předpověď. Liší se jen počet vstupů sítě
> (počet příznaků, nebo délka okna) a třída `Dataset`, která síti data
> podává. Tréninková smyčka je v obou případech stejná.

> **Balíčky `dataio/` a `utils/`, třída `ResidualAnalyzer` a většina třídy
> `Trainer` jsou předvyplněny.** Logování nepíšete, pouze pozorujete jeho
> výstup v konzoli a v souboru `logs/trainer.log`.

> **Tři balíčky, tři role.** `src/` je **látka cvičení**: model, učení,
> datasety a metriky, které programujete. `dataio/` přivádí **data a nastavení
> dovnitř a výsledky ven** (načtení, rozdělení, konfigurace, grafy). `utils/`
> je **orchestrace pipeline**: hlášky, přepravky výsledků mezi fázemi a učení
> s kontrolou průběhu. Závislosti vedou jen jedním směrem, `utils/` používá
> `src/` i `dataio/`, nikdy naopak. Vstupní skript `cviceni_11.py` tak
> obsahuje jen scénář cvičení: fáze A1–B4 v pořadí, v jakém běží.

---

## Instalace a spuštění

### 1. Vytvoření virtuálního prostředí

```bash
python -m venv .venv
```

Aktivace (Windows):
```bash
.venv\Scripts\activate
```

Aktivace (Linux / macOS):
```bash
source .venv/bin/activate
```

### 2. Instalace závislostí

```bash
pip install -r requirements.txt
```

> **PyTorch je velká závislost.** Instaluje se varianta **pro procesor (CPU)**,
> která má přibližně 200 MB, a stažení proto chvíli trvá. Grafickou kartu
> cvičení nepotřebuje, sítě jsou malé a učení trvá desítky sekund.
> `requirements.txt` začíná řádkem `--extra-index-url`, který na Windows a na
> Linuxu vybere menší CPU sestavení místo sestavení s podporou CUDA (několik
> GB). Tento řádek neodstraňujte.

### 3. Spuštění

```bash
python cviceni_11.py
```

Pipeline má dvě části a **každá fáze má vlastní ošetření chyb**:

| Fáze | Obsah | Co potřebuje hotové |
|:---|:---|:---|
| A1 | data: sinusovka a diabetes (rozdělení, standardizace) | nic (předvyplněno) |
| A2 | sinusovka: učení sítě, proložení křivky, křivka učení | `MLP`, `Trainer._train_step`; pro výpis metrik `mse`, `mae`, `r2` |
| A3 | diabetes: učení sítě, křivka učení | `MLP`, `Trainer._train_step` |
| A4 | diabetes: MSE, MAE a $R^2$ na testovacích datech | úspěšná fáze A3, `mse`, `mae`, `r2` |
| A5 | diabetes: reziduální diagnostika (čtyři grafy, Shapirův–Wilkův test) | úspěšná fáze A3 (diagnostika je předvyplněna) |
| A6 | uložení modelu do `models/` a kontrolní načtení | úspěšná fáze A3, `Trainer.save`, `Trainer.load` |
| B1 | data: řada teplot, chronologické rozdělení, standardizace | nic (předvyplněno) |
| B2 | okénkování a učení sítě | `WindowedDataset`, `MLP`, `Trainer._train_step` |
| B3 | MAE a RMSE modelu proti persistence baseline | úspěšná fáze B2, `mse`, `mae`, `persistence_baseline` |
| B4 | graf předpovědi v čase a reziduí v čase | úspěšná fáze B2 |

Dokud nejsou úkoly hotové, fáze, která narazí na nedokončenou část, skončí
hláškou `[NENI HOTOVO] Úkol: …` a pipeline **pokračuje další fází**. Selže-li
již doplněná část (nesplněný assert, nesouhlasící tvary tenzorů), fáze skončí
hláškou `[CHYBA IMPLEMENTACE]` s popisem výjimky. Nikdy nedostanete holý
traceback. Fáze, která závisí na neúspěšné předchozí fázi, se ohlásí jako
`[PRESKOCENO]` s uvedením důvodu.

Další hlášky, na které můžete narazit:

| Hláška | Význam |
|:---|:---|
| `[CHYBA UCENI]` | ztráta během učení divergovala (nekonečná nebo prudce rostoucí); zmenšete `lr` |
| `[POZOR]` | výsledek je podezřelý, ale pipeline pokračuje: síť se přeučuje, trénovací ztráta neklesá, síť neporazila persistence baseline, nebo má baseline nulovou chybu (chyba o jedna) |
| `[CHYBA DAT]` | data nelze načíst nebo rozdělit (chybějící či poškozený soubor s teplotami, nevhodné rozdělení); příslušná část se přeskočí |
| `[CHYBA KONFIGURACE]` | nesmyslná hodnota v `config.yaml` |

> **Chybějící soubor s teplotami.** Soubor `data/daily_min_temperatures.csv`
> je součástí repozitáře. Pokud by chyběl nebo byl poškozený, fáze B1 to
> oznámí hláškou `[CHYBA DAT]` s číslem problematického řádku, část B se
> přeskočí a část A proběhne celá.

> **Konfigurace se kontroluje hned na začátku.** Nesmyslná hodnota nebo
> syntaktická chyba v `config.yaml` ukončí pipeline hláškou
> `[CHYBA KONFIGURACE]` se jménem klíče. Výjimkou je `forecast.window_size`
> delší než trénovací nebo testovací část řady: to se pozná až po rozdělení
> dat a fáze B1 přeskočí jen část B.

S referenční implementací a výchozí konfigurací trvá celá pipeline na běžném
notebooku jednu až dvě minuty podle výkonu počítače (nejdéle fáze B2) a dává
tyto výsledky
na testovacích datech:

| Úloha | Výsledek |
|:---|:---|
| sinusovka (A2) | $R^2 \approx 0{,}97$, MAE $\approx 0{,}10$ |
| diabetes (A4) | $R^2 \approx 0{,}25$, MAE $\approx 54$, RMSE $\approx 65$ |
| teploty, síť (B3) | MAE $\approx 1{,}73$ °C, RMSE $\approx 2{,}21$ °C |
| teploty, persistence baseline (B3) | MAE $\approx 1{,}95$ °C, RMSE $\approx 2{,}48$ °C |

Vaše čísla se mohou v posledních číslicích lišit podle verze knihoven
a procesoru.

> **Nízké $R^2$ u diabetu není chyba ve vašem kódu.** Při výchozích 500
> epochách se síť **přeučí**. Na křivce učení
> (`graphs/diabetes_krivka_uceni.png`) trénovací ztráta stále klesá, kdežto
> validační přestane klesat už po zhruba 20 epochách, dalších asi 150 epoch se
> drží na stejné úrovni a potom roste. Pipeline na to ve fázi A3 upozorní. Se
> `regression.epochs: 30` vychází $R^2 \approx 0{,}40$ a MAE $\approx 50$,
> o něco lépe než lineární regrese na stejných datech ($R^2 \approx 0{,}39$).
> Dataset má jen 442 pacientů a i dobré modely na něm vysvětlí nejvýše zhruba
> polovinu rozptylu progrese, výrazně vyšší hodnoty proto čekat nelze.
> Sinusovce naopak 30 epoch nestačí. Obě úlohy sdílejí sekci `regression`,
> takže počet epoch je kompromis. Srovnání architektur a výzva k jejich
> překonání jsou v [Pokynech k vypracování](#výzva-navrhněte-lepší-síť).

---

## Teoretický základ

### 1. Z NumPy do PyTorche

V Cvičení 08–10 jste napsali vše, co síť k učení potřebuje. PyTorch tytéž
kroky provádí za vás. Nejde tedy o novou látku, ale o jiný zápis látky známé:

| Cvičení 10 (ruční NumPy) | Cvičení 11 (PyTorch) |
|:---|:---|
| pole `np.ndarray` | tenzor `torch.Tensor` |
| `Linear` (`x @ W + b`), `Neuron(Linear, Activation)` | `nn.Linear`, `nn.Tanh` |
| `Sequential([...])` s ručním `forward` | `nn.Sequential(...)`, volání `self.net(x)` |
| inicializátor vah (`XavierInit`) | výchozí inicializace `nn.Linear` (v Cvičení 10 `PyTorchDefaultInit`) |
| `MSE.forward(prediction, target)` | `nn.MSELoss()(y_pred, y_batch)` |
| `Loss.gradient` + `Activation.derivative` + `Sequential.backward` | `loss.backward()` (autograd) |
| `Linear.update`: `W = W - learning_rate * dW` | `optimizer.step()` s `torch.optim.SGD` |
| míchání a dávky přes `rng.permutation` v `Trainer.run` | `Dataset` + `DataLoader` |
| `np.savez` vah po vrstvách | `torch.save(model.state_dict(), path)` |

**Autograd.** Tenzor, který je parametrem modelu, si pamatuje, z jakých
operací vznikly hodnoty, které se z něj počítají. Během dopředného průchodu
tak PyTorch sestavuje **výpočetní graf**. Volání `loss.backward()` tento graf
projde od ztráty zpět a ke každému parametru uloží do atributu `.grad`
derivaci ztráty podle něj. Je to přesně řetězové pravidlo, které jste
v Cvičení 10 psali po vrstvách: mezistavy `io_` a `z_` jsou uloženy v grafu
a derivace jednotlivých operací zná knihovna.

> **Nejde o „zapomeňte vše".** Autograd nedělá nic, čemu byste nerozuměli.
> Znáte stroj zevnitř a nyní ho jen řídíte. Když se síť neučí, hledáte příčinu
> ve stejných místech jako dřív: krok učení, měřítko vstupů, saturace, tvary.

### 2. Stavební bloky PyTorche

**Model je potomek `nn.Module`.** V konstruktoru vytvoří vrstvy a uloží je do
atributů, v metodě `forward` popíše dopředný průchod. `nn.Module` si
zaregistruje všechny parametry vrstev, takže je optimalizátor najde přes
`model.parameters()`. Model se volá jako funkce, `model(x)`, což uvnitř
zavolá `forward`.

```python
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(...)     # Linear, Tanh, ..., Linear

    def forward(self, x):
        return self.net(x)
```

**`nn.Linear(n_in, n_out)`** počítá $\mathbf{x}W^{\top} + \mathbf{b}$ pro dávku
tvaru `(B, n_in)` a vrací `(B, n_out)`. Řádek je vzorek, stejně jako
v předchozích cvičeních. Matici vah ukládá PyTorch transponovaně, ve tvaru
`(n_out, n_in)`, na použití se tím nic nemění.

**Optimalizátor `torch.optim.SGD(model.parameters(), lr=...)`** je gradientní
sestup z Cvičení 10: `optimizer.step()` provede pro každý parametr
$\theta \leftarrow \theta - \eta\,\partial L/\partial\theta$. Jiné
optimalizátory jsou tématem samostatného doplňkového materiálu.

**`Dataset` a `DataLoader`.** `Dataset` je objekt se dvěma metodami:
`__len__` vrací počet vzorků a `__getitem__(idx)` vrací jeden vzorek jako
dvojici tenzorů `(x, y)`. `DataLoader` z něj skládá dávky a podle parametru
`shuffle` vzorky před každou epochou zamíchá. Nahrazuje tak ruční dávkování
z `Trainer.run` v Cvičení 10.

**Tvary v tomto cvičení** jsou jednotné pro obě části:

| Objekt | Tvar | Poznámka |
|:---|:---|:---|
| vzorek z datasetu, `x` | `(n_features,)` | u předpovědi `(window_size,)` |
| vzorek z datasetu, `y` | `(1,)` | cíl je vektor délky 1, ne skalár |
| dávka `x_batch` | `(B, n_features)` | |
| dávka `y_batch` | `(B, 1)` | |
| výstup modelu `y_pred` | `(B, 1)` | stejný tvar jako `y_batch` |

Všechny tenzory jsou typu `torch.float32`.

**Stav `train()` a `eval()`.** V Cvičení 10 jste tento přepínač poznali jako
oprávnění měnit váhy. V PyTorchi má i věcný důvod: některé vrstvy (dropout,
normalizace dávkou) se při učení chovají jinak než při vyhodnocení. Naše síť
takové vrstvy nemá, přesto je správné přepínat vždy. Při vyhodnocení se navíc
používá blok `with torch.no_grad():`, který vypne stavbu výpočetního grafu.
Gradienty tam nejsou potřeba a výpočet je rychlejší.

### 3. Tréninkový krok

Jeden krok učení na jedné dávce má v PyTorchi vždy tuto podobu:

```python
optimizer.zero_grad()              # 1. vynuluj gradienty z minulého kroku
y_pred = model(x_batch)            # 2. dopředný průchod (staví se výpočetní graf)
loss = loss_fn(y_pred, y_batch)    # 3. ztráta
loss.backward()                    # 4. zpětný průchod: naplní .grad všech parametrů
optimizer.step()                   # 5. posun parametrů proti gradientu
```

Srovnání s krokem z Cvičení 10:

| Cvičení 10 | Cvičení 11 |
|:---|:---|
| `prediction = model.forward(x_batch)` | `y_pred = model(x_batch)` |
| `loss.forward(prediction, y_batch)` | `loss = loss_fn(y_pred, y_batch)` |
| `gradients = model.backward(loss.gradient(prediction, y_batch))` | `loss.backward()` |
| `model.update(gradients, learning_rate)` | `optimizer.step()` |
| — | `optimizer.zero_grad()` |

> **Proč `zero_grad`.** PyTorch gradienty do `.grad` **přičítá**, nepřepisuje.
> Bez vynulování by se v každém kroku sčítaly gradienty všech dosavadních
> dávek a učení by se rozpadlo. V Cvičení 10 tento krok nebyl potřeba, protože
> `backward` vracel pokaždé nový seznam gradientů.

> **Pozor na tvary.** `nn.MSELoss` porovnává tenzory prvek po prvku. Má-li
> `y_pred` tvar `(B, 1)` a `y_batch` tvar `(B,)`, PyTorch je podle pravidel
> broadcastingu rozšíří na `(B, B)` a spočítá průměr přes všech $B^2$ dvojic.
> Program nespadne, vypíše jen varování, a síť se učí předpovídat průměr.
> Proto mají cíle v celém cvičení tvar `(B, 1)` a tréninkový krok shodu tvarů
> ověřuje. Stejná past existuje v NumPy u metrik: `(n,)` minus `(n, 1)` dá
> matici `(n, n)`.

### 4. Regrese: lineární výstup, ztráta a metriky

V Cvičení 10 končila síť sigmoidou, protože výstup byl pravděpodobnost třídy
z intervalu $(0, 1)$. Regresní síť předpovídá **libovolné reálné číslo**,
proto poslední vrstva **nemá žádnou aktivaci**. Sigmoida nebo tanh na výstupu
by omezily rozsah předpovědí a hodnoty mimo něj by síť nikdy nevrátila.
Skryté vrstvy aktivaci mít musí (zde tanh), jinak by se síť zhroutila do
jediné lineární vrstvy (Cvičení 09).

Ztrátou je **střední kvadratická chyba**. Pro skutečné hodnoty $y_i$,
předpovědi $\hat{y}_i$ a $n$ vzorků:

| Metrika | Definice | Jednotka | Vlastnosti |
|:---|:---|:---|:---|
| MSE | $\dfrac{1}{n}\sum_i (y_i - \hat{y}_i)^2$ | čtverec jednotky cíle | silně trestá velké chyby, citlivá na odlehlé hodnoty |
| RMSE | $\sqrt{\text{MSE}}$ | jednotka cíle | totéž, ale ve srozumitelné jednotce |
| MAE | $\dfrac{1}{n}\sum_i \lvert y_i - \hat{y}_i\rvert$ | jednotka cíle | typická velikost chyby, robustnější vůči odlehlým hodnotám |
| $R^2$ | $1 - \dfrac{SS_{\text{res}}}{SS_{\text{tot}}}$ | bez jednotky | podíl rozptylu cíle vysvětlený modelem |

kde

$$SS_{\text{res}} = \sum_i (y_i - \hat{y}_i)^2, \qquad SS_{\text{tot}} = \sum_i (y_i - \bar{y})^2, \qquad \bar{y} = \frac{1}{n}\sum_i y_i .$$

$SS_{\text{tot}}$ je chyba nejjednoduššího možného modelu, který vždy
předpoví průměr $\bar{y}$. Koeficient determinace tedy srovnává model
s tímto průměrem:

| Hodnota $R^2$ | Význam |
|:---:|:---|
| $1$ | dokonalá předpověď |
| $0$ | model je stejně dobrý jako předpověď průměrem |
| $< 0$ | model je horší než předpověď průměrem |

> **$R^2$ není čtverec korelace.** Název svádí k představě, že $R^2$ leží
> mezi 0 a 1. To platí jen pro lineární regresi vyhodnocenou na trénovacích
> datech. Na testovacích datech a pro obecný model může být $R^2$ záporné.

V tomto cvičení metriky **programujete sami v NumPy**. Knihovna
`sklearn.metrics` slouží v testech jen jako reference, se kterou se vaše
výsledky porovnávají.

**Standardizace vstupů i cíle.** Stejně jako v Cvičení 10 se příznaky převádějí
na nulový průměr a jednotkovou směrodatnou odchylku, statistikami spočtenými
**jen z trénovacích dat**. U regrese se standardizuje i **cíl**: progrese
diabetu má hodnoty kolem 150 a chyba v řádu stovek by při kroku učení 0,01
vedla k obrovským gradientům a saturaci tanh. Síť se proto učí standardizovaný
cíl a předpovědi se před výpočtem metrik převedou zpět:
$\hat{y} = \hat{y}_{\text{std}}\cdot s_y + \bar{y}$. MSE a MAE tak vycházejí
v původních jednotkách.

### 5. Část A: reziduální diagnostika

Jediné číslo, ať MSE nebo $R^2$, neřekne, **jak** model chybuje. To ukážou
**rezidua** $e_i = y_i - \hat{y}_i$. Vystihuje-li model data dobře, zbyl
v reziduích jen náhodný šum: nulový průměr, stálý rozptyl, žádná závislost na
předpovědi ani na příznacích, přibližně normální rozdělení. Třída
`ResidualAnalyzer` nabízí pět pohledů:

| Metoda | Co ukazuje | V pořádku | Varovný signál |
|:---|:---|:---|:---|
| `histogram_residuals` | rozdělení reziduí s proloženou normální křivkou | symetrický zvon kolem nuly | posun od nuly, šikmost, dva vrcholy |
| `qq_residuals` | kvantily reziduí proti kvantilům normálního rozdělení | body na přímce | odklon na koncích (těžké chvosty), esovitý tvar |
| `residuals_vs_prediction` | rezidua proti $\hat{y}$, pás $\pm 2\sigma$ | vodorovný pás bez struktury | trychtýř (rozptyl roste s $\hat{y}$), oblouk |
| `residuals_vs_x` | rezidua proti jednomu příznaku | bez závislosti | oblouk nebo trend: model vliv příznaku nevystihl |
| `residual_normality` | Shapirův–Wilkův test normality | $p > 0{,}05$ | $p \le 0{,}05$: normalita zamítnuta |

> **Trychtýř a oblouk.** Rozšiřující se pás v grafu reziduí proti předpovědi
> se nazývá **heteroskedasticita**: model je pro některé hodnoty spolehlivější
> než pro jiné. Oblouk znamená **nezachycenou nelinearitu**: model systematicky
> podhodnocuje v jedné oblasti a nadhodnocuje v jiné.

> **Shapirův–Wilkův test čtěte s rozmyslem.** Nulová hypotéza je „rezidua
> pocházejí z normálního rozdělení". Malá $p$-hodnota ji zamítá. Na velkých
> souborech test zamítne normalitu i kvůli nepatrné odchylce, která nemá
> praktický význam. Vždy jej proto čtěte společně s QQ grafem.

Rozdělení dat v části A je **náhodné**. Pacienti jsou navzájem nezávislí,
pořadí řádků v tabulce nenese žádnou informaci a zamíchání je v pořádku.
V části B to neplatí.

### 6. Část B: předpověď časové řady

Časová řada je posloupnost hodnot $y_1, y_2, \dots, y_T$ měřených v pravidelných
okamžicích. Úloha: z posledních $k$ hodnot předpovědět následující,

$$\hat{y}_t = f(y_{t-k}, y_{t-k+1}, \dots, y_{t-1}).$$

Na regresi se tato úloha převede **okénkováním**. Po řadě se posouvá okno
délky $k$ (`window_size`). Hodnoty v okně jsou vstupem sítě a hodnota hned za
oknem je cílem:

```
řada:      10   11   12   13   14   15        window_size = 3

vzorek 0: [10   11   12] → 13
vzorek 1:      [11   12   13] → 14
vzorek 2:           [12   13   14] → 15
```

Z řady délky $T$ vznikne $T - k$ vzorků. Vzorek s indexem `idx` je

```python
x = series[idx : idx + window_size]      # posledních k známých hodnot
y = series[idx + window_size]            # hodnota, která následuje
```

> **Okno se posouvá po hodnotách signálu, ne po časové ose.** Vstupem sítě
> jsou **minulé hodnoty řady**. Starší verze tohoto cvičení okénkovala
> časovou osu, síť tak dostávala čísla dnů místo teplot a učila se funkci
> času, nikoli závislost na minulosti. Taková síť neumí předpovídat nic, co
> v trénovacím období neviděla. Třída `WindowedDataset`, kterou zde píšete,
> tuto chybu opravuje.

**Chronologické rozdělení.** Řada se rozdělí v jediném bodě: prvních 80 %
hodnot slouží k učení, zbylých 20 % k testování.

```
|<------------- trénovací část ------------->|<-- testovací část -->|
 1981                                      1988                  1990
```

> **Zamíchání časové řady je únik dat.** Kdyby se okna před rozdělením náhodně
> zamíchala, trénovací sada by obsahovala okna z let 1989 a 1990 a testovací
> okna z roku 1985. Sousední okna se navíc téměř celá překrývají, takže
> k většině testovacích oken by v trénovací sadě existovalo okno posunuté
> o jediný den. Model by se „učil na budoucnosti" a testovací chyba by byla
> nerealisticky nízká. Je to **leakage z Cvičení 06 v časové podobě**: tam
> unikala informace z testovacích dat do předzpracování, zde uniká budoucnost
> do minulosti. Při nasazení přitom model budoucnost nikdy nevidí.

> **Co se míchat smí.** Pořadí **trénovacích oken v dávkách** se míchat může
> (`DataLoader(..., shuffle=True)`). Každé okno drží své hodnoty ve správném
> pořadí a všechna trénovací okna leží před dělicím bodem, takže do učení
> žádná budoucnost neproniká. Zakázáno je míchat **před rozdělením**, tedy
> rozhodovat náhodně o tom, které okno bude trénovací a které testovací.

Standardizace řady používá průměr a směrodatnou odchylku **jen z trénovací
části**, ze stejného důvodu.

### 7. Persistence baseline: laťka, kterou je třeba překonat

Nejjednodušší předpověď časové řady nepotřebuje žádný model:

$$\hat{y}_t^{\text{pers}} = y_{t-1} \qquad \text{(„zítra bude jako dnes").}$$

U řad, které se mění pomalu, je tato naivní předpověď překvapivě dobrá.
Teplota dvou po sobě jdoucích dnů se obvykle liší jen o několik stupňů. Model,
který dosáhne MAE 2 °C, vypadá působivě do chvíle, než zjistíme, že naivní
předpověď má MAE podobné.

> **Model má cenu jen tehdy, když persistence porazí.** Je to táž myšlenka
> jako u nevyvážených tříd v Cvičení 06: přesnost 95 % nic neznamená, pokud
> 95 % vzorků patří do jedné třídy a klasifikátor, který vždy odpoví touto
> třídou, dosáhne téhož. Každou metriku je třeba vztáhnout k tomu, čeho lze
> dosáhnout bez přemýšlení.

**Zarovnání.** Aby se chyby modelu a baseline daly porovnat, musí obě
předpovědi patřit **týmž cílům**. Cílem vzorku `i` z `WindowedDataset` je
`series[i + window_size]` a naivní předpovědí pro něj je hodnota řady
o jeden den dříve. Funkce `persistence_baseline(series, window_size)` vrací
tyto předpovědi pro všechny cíle najednou, tedy pole tvaru
`(len(series) - window_size,)`:

```
řada:       10   11   12   13   14   15        window_size = 3
cíle:                      13   14   15        (cíle všech oken)
baseline:                  12   13   14        (pod každým cílem hodnota z předchozího dne)
```

Prvek `i` baseline patří cíli `i`. Jakým řezem pole `series` baseline
získat, je předmětem úkolu (Blok VI).

> **Chyba o jedna.** Posun o jeden prvek na kteroukoli stranu je nejčastější
> chybou. Vrátí-li funkce samotné cíle, má baseline nulovou chybu a žádný
> model ji neporazí. Vrátí-li hodnoty o dva dny starší („zítra bude jako
> předevčírem"), je baseline zbytečně špatná a model vedle ní vypadá lépe,
> než jaký je. Obojí odhalí test `TestPersistenceBaseline`. Správnost si
> ověřte na malém příkladu výše dřív, než funkci použijete na skutečná data.

### 8. Hodnocení předpovědi

Předpověď se hodnotí třemi způsoby:

1. **MAE a RMSE** modelu na testovací části, v původních jednotkách (°C).
2. **Tytéž metriky pro persistence baseline** a jejich poměr. Pipeline
   vypisuje, o kolik procent je model lepší nebo horší.
3. **Rezidua v čase.** Graf reziduí proti indexu dne ukáže, zda v chybě
   nezůstala systematická složka: sezónní vlna (model nevystihl roční chod),
   úseky s trvale kladnou nebo zápornou chybou, rostoucí rozptyl.

$R^2$ se u časových řad s výrazným sezónním chodem nepoužívá jako hlavní
metrika. Roční chod teploty tvoří většinu rozptylu a vysokého $R^2$ dosáhne
i persistence baseline. Rozhodující je srovnání s ní.

> **Předpověď o jeden krok.** Síť předpovídá vždy jen následující den a na
> vstupu má skutečně naměřené hodnoty. Předpověď na více dnů dopředu, při níž
> se na vstup vracejí vlastní předpovědi sítě, je těžší úloha a chyba v ní
> s délkou horizontu roste.

### 9. Uložení modelu: `state_dict`

Naučený model je určen svými parametry. PyTorch je vrací jako **`state_dict`**,
slovník, který jménům vrstev přiřazuje tenzory vah a biasů:

```python
torch.save(model.state_dict(), "models/mlp_diabetes.pt")      # uložení

model = MLP(input_dim=10, hidden_dims=[32, 16])               # stejná architektura
model.load_state_dict(torch.load("models/mlp_diabetes.pt", weights_only=True))
```

Navazuje to na `save`/`load` z Cvičení 05–10: ukládá se **stav**, nikoli
objekt. Při načítání je proto nejprve třeba vytvořit model se stejnou
architekturou a parametry do něj nahrát.

> **Neukládejte celý model.** `torch.save(model, path)` serializuje celý
> objekt modulem `pickle`. Soubor pak závisí na přesné podobě zdrojového
> kódu třídy a jeho načtení může spustit libovolný kód, což je bezpečnostní
> riziko. Parametr `weights_only=True` u `torch.load` zajistí, že se načtou
> pouze tenzory.

### 10. Logování

Stejně jako v Cvičení 10 zapisuje `Trainer` průběh učení modulem `logging`:
do konzole milníky na úrovni `INFO` a do rotujícího souboru `logs/trainer.log`
záznam každé epochy na úrovni `DEBUG`. Do logu patří jen skaláry (ztráty,
počty), nikdy celé tenzory ani jednotlivé vzorky. Logy se do repozitáře
neukládají (`logs/*.log` je v `.gitignore`).

---

## Konfigurace projektu

> **Vše nastavitelné je v `config.yaml`.** Kód cvičení neobsahuje žádné
> „magické" konstanty: cesty ke vstupům a výstupům, parametry dat, sítí
> a učení i prahy kontrol, které pipeline vypisuje, se čtou z konfigurace.
> Chcete-li cokoli vyzkoušet jinak, měníte `config.yaml`, nikoli kód.
> V kódu zůstávají jen jména výstupních souborů (např. `sinus_prolozeni.png`).

### Soubor `config.yaml`

```yaml
paths:                        # vstupy a výstupy, relativně ke kořeni repozitáře
  temperature_csv: data/daily_min_temperatures.csv   # řada denních teplot (část B)
  graphs_dir: graphs          # výstupní grafy (.png)
  models_dir: models          # uložené modely (state_dict, .pt)
  logs_dir: logs              # log učení (trainer.log)

sinusoid:                     # data části A: zašuměná sinusovka na intervalu [-π, π]
  n_samples: 300              # počet bodů
  noise: 0.1                  # směrodatná odchylka gaussovského šumu

regression:                 # část A: regrese (sinusovka + diabetes)
  hidden_dims: [32, 16]     # skryté vrstvy; vstup a lineární výstup (1) se dopočítají
  lr: 0.01                  # krok učení optimalizátoru SGD
  epochs: 500               # počet průchodů trénovacími daty
  batch_size: 32            # velikost dávky v DataLoaderu
  test_size: 0.2            # podíl testovacích vzorků (NÁHODNÉ rozdělení)

forecast:                   # část B: předpověď časové řady
  window_size: 14           # délka okna: kolik posledních hodnot předpovídá další
  hidden_dims: [32, 16]     # skryté vrstvy; vstup má window_size hodnot
  lr: 0.01                  # krok učení optimalizátoru SGD
  epochs: 300               # počet průchodů trénovacími okny
  batch_size: 32            # velikost dávky v DataLoaderu
  train_frac: 0.8           # podíl řady na učení (CHRONOLOGICKÉ rozdělení)

diagnostics:                  # prahy kontrol, které pipeline vypisuje
  divergence_factor: 10.0     # divergence: konečná trénovací ztráta > 10× ztráta 1. epochy
  overfit_rise: 0.10          # přeučení: konečná validační ztráta o více než 10 % nad minimem ...
  overfit_min_fraction: 0.9   # ... a minimum nastalo před 90 % epoch
  load_tolerance: 1.0e-6      # max. rozdíl predikcí původního a načteného modelu
  zero_error_tolerance: 1.0e-12  # baseline s nulovou chybou = vrací samotné cíle
  normality_alpha: 0.05       # hladina významnosti Shapirova–Wilkova testu

seed: 42                      # seed rozdělení dat, inicializace vah i míchání dávek (>= 0)
```

Sekce `diagnostics` nemění učení ani výsledky, jen to, kdy pipeline ohlásí
`[POZOR]` nebo `[CHYBA UCENI]`. Například s `overfit_rise: 1.0` se varování
o přeučení objeví, až když validační ztráta vzroste na dvojnásobek svého
minima.

### Typovaná konfigurace (dataclassy)

```
ExperimentConfig
├── paths:       PathsConfig(temperature_csv, graphs_dir, models_dir, logs_dir)
├── sinusoid:    SinusoidConfig(n_samples, noise)
├── regression:  RegressionConfig(hidden_dims, test_size, training)
│                  └── training: TrainingConfig(lr, epochs, batch_size, seed)
├── forecast:    ForecastConfig(window_size, hidden_dims, train_frac, training)
│                  └── training: TrainingConfig(lr, epochs, batch_size, seed)
├── diagnostics: DiagnosticsConfig(divergence_factor, overfit_rise, overfit_min_fraction,
│                                  load_tolerance, zero_error_tolerance, normality_alpha)
└── seed:        int
```

Dataclassy konfigurace a funkce `load_config` a `validate_config` žijí
v `dataio/config_manager.py`.

Klíče `lr`, `epochs` a `batch_size` dané sekce se spolu s kořenovým `seed`
složí do `TrainingConfig`. Ten se předává třídě `Trainer`, která díky tomu
nemusí vědět, kterou z obou úloh právě učí.

K hodnotám se přistupuje **přes atributy, nikdy přes klíče slovníku**:

```python
# Místo:   cfg["forecast"]["window_size"]   ← chyba až za běhu při překlepu
# Správně: cfg.forecast.window_size          ← editor odhalí překlep okamžitě
```

`validate_config()` ověří v sekcích `regression` a `forecast`, že `lr > 0`,
`epochs >= 1`, `batch_size >= 1` a `hidden_dims` je neprázdný seznam kladných
celých čísel, a dále že `0 < test_size < 1`, `window_size >= 1`,
`0 < train_frac < 1` a `seed >= 0`. V ostatních sekcích kontroluje, že cesty
jsou neprázdné, `n_samples >= 10`, `noise >= 0`, `divergence_factor > 1`,
`overfit_rise > 0`, `0 < overfit_min_fraction <= 1`, tolerance jsou nezáporné
a `0 < normality_alpha < 1`. Hlídá i typy (například `epochs: 10.7` nebo
`hidden_dims: 32` jsou chyba). Při porušení vyhodí `ValueError` se
srozumitelnou hláškou, která jmenuje klíč i zadanou hodnotu.

> **Síť je záměrně malá.** Dvě skryté vrstvy s 32 a 16 neurony mají pro
> sinusovku 609 parametrů a na obě úlohy stačí. Starší verze cvičení
> prokládala jednorozměrnou sinusovku sítí se šesti vrstvami po 70 neuronech.
> Taková síť má přes 25 000 parametrů, sinusovku proloží stejně dobře jako
> výchozí síť se 609 parametry a u diabetu se přeučí nejvíc ze všech
> zkoušených. Větší síť není lepší síť. Srovnání architektur najdete
> v [Pokynech k vypracování](#výzva-navrhněte-lepší-síť).

---

## Pokyny k vypracování

Pracujte **v tomto pořadí**. Po každém bloku můžete spustit `pytest -v`
a sledovat, jak ubývá `xfail`, a `python cviceni_11.py`, kde přibývají
dokončené fáze.

### Blok I: `MLP` v `src/model.py`

```
# __init__(input_dim, hidden_dims, output_dim):
# 1. Ověřte (assert), že input_dim >= 1, output_dim >= 1 a hidden_dims je neprázdný.
# 2. Projděte hidden_dims a do seznamu vrstev přidávejte dvojice
#        nn.Linear(předchozí_šířka, h),  nn.Tanh()
#    předchozí šířka začíná na input_dim a po každé vrstvě je h.
# 3. Na konec přidejte nn.Linear(předchozí_šířka, output_dim) BEZ aktivace.
# 4. self.net = nn.Sequential(*vrstvy)
#
# forward(x):  vraťte self.net(x)
```

Pro `hidden_dims = [32, 16]` a `input_dim = 10` vznikne
`Linear(10, 32) → Tanh → Linear(32, 16) → Tanh → Linear(16, 1)`. Jméno
atributu `self.net` dodržte, testy jej používají.

### Blok II: `Trainer._train_step` v `src/trainer.py` (jádro cvičení)

Konstruktor už připravil `self.model`, `self.optimizer` (SGD) a `self.loss_fn`
(MSE). Doplňte jeden krok učení podle odd. 3 teorie:

```
# 1. Vynulujte gradienty optimalizátoru.
# 2. Spočítejte predikci modelu pro x_batch.
# 3. Ověřte (assert), že predikce a y_batch mají stejný tvar.
# 4. Spočítejte ztrátu self.loss_fn(predikce, y_batch).
# 5. Zavolejte zpětný průchod ztráty.
# 6. Proveďte krok optimalizátoru.
# 7. Vraťte hodnotu ztráty jako Python float (loss.item()).
```

Metody `fit`, `_evaluate` a `predict` jsou předvyplněny. Přečtěte si je:
`fit` volá váš `_train_step` pro každou dávku z `DataLoader` a přepíná model
mezi `train()` a `eval()`. Po dokončení bloků I a II proběhnou fáze A2 a A3
a v `logs/trainer.log` najdete záznam každé epochy.

### Blok III: `mse`, `mae`, `r2` v `src/metrics.py`

Čisté NumPy, bez `sklearn`. Vzorce jsou v odd. 4 teorie.

```
# Ve všech třech funkcích:
# 1. Ověřte (assert), že y_true a y_pred mají stejný tvar.
# mse:  průměr čtverců rozdílů
# mae:  průměr absolutních hodnot rozdílů
# r2:   ss_res = součet čtverců (y_true - y_pred)
#       ss_tot = součet čtverců (y_true - průměr y_true)
#       ověřte (assert), že ss_tot > 0
#       vraťte 1 - ss_res / ss_tot
# Vždy vracejte Python float.
```

Testy `TestMetriky` porovnají vaše výsledky se `sklearn.metrics`. Po tomto
bloku fáze A2 a A4 vypíšou metriky a fáze A5 vykreslí reziduální diagnostiku.
Prohlédněte si grafy v `graphs/` a interpretujte je podle tabulky v odd. 5.

### Blok IV: `Trainer.save` a `Trainer.load` v `src/trainer.py`

```
# save(path):
#     uložte self.model.state_dict() funkcí torch.save
#
# load(path, model, config):      (classmethod)
#     1. načtěte state_dict funkcí torch.load(path, weights_only=True)
#     2. nahrajte jej do předaného modelu metodou load_state_dict
#     3. vraťte novou instanci cls(model, config)
```

Fáze A6 pak musí vypsat `[OK]` s nulovým rozdílem predikcí původního
a načteného modelu.

### Blok V: `WindowedDataset` v `src/datasets.py`

Vzorem je hotová třída `TabularDataset` ve stejném souboru. Postup podle
odd. 6 teorie:

```
# __init__(series, window_size):
# 1. Převeďte series na pole typu float32 (np.asarray).
# 2. Ověřte (assert), že řada je 1D, že window_size >= 1 a že řada je delší
#    než window_size.
# 3. Uložte self.series jako tenzor (torch.as_tensor) a self.window_size.
#
# __len__():        počet oken = délka řady - window_size
#
# __getitem__(idx):
#     x = self.series[idx : idx + self.window_size]            tvar (window_size,)
#     y = self.series[idx + self.window_size].reshape(1)        tvar (1,)
#     vraťte (x, y)
```

### Blok VI: `persistence_baseline` v `src/metrics.py`

```
# 1. Převeďte series na pole typu float64.
# 2. Ověřte (assert), že řada je 1D, že window_size >= 1 a že řada je delší
#    než window_size.
# 3. Vraťte řez řady, jehož prvek i je poslední známá hodnota před cílem
#    series[window_size + i]. Výsledek má tvar (len(series) - window_size,).
```

Řez si nejprve odvoďte na příkladu z odd. 7 (řada `10 … 15`, okno 3) a ověřte,
že vyjde `[12, 13, 14]`. Po dokončení
bloků V a VI proběhne celá část B a fáze B3 vypíše srovnání modelu
s baseline.

### Doporučené experimenty (nehodnotí se)

Měňte **pouze** `config.yaml` a pozorujte grafy v `graphs/` a log:

1. `forecast.window_size`: `1`, `7`, `30`. Jak se mění náskok modelu před
   persistence baseline? Čemu se model s oknem délky 1 může nanejvýš naučit?
2. `regression.lr`: `0.1` a `0.001`. Kdy učení osciluje a kdy je příliš
   pomalé? Porovnejte křivky učení sinusovky.
3. `regression.hidden_dims`: `[4]` a `[64, 64]`. Jak se změní proložení
   sinusovky a rozdíl mezi trénovací a validační ztrátou u diabetu?
4. `regression.epochs`: `30` a `2000`. Sledujte u diabetu křivku učení a $R^2$
   na testovacích datech. Od které epochy validační ztráta přestává klesat,
   zatímco trénovací klesá dál? Co se při 30 epochách stane se sinusovkou
   a proč jí krátké učení nestačí?
5. `forecast.train_frac`: `0.5`. Stačí síti pět let dat?

### Výzva: navrhněte lepší síť

Výchozí architektura není to nejlepší, co lze najít. Následující tabulka
shrnuje, co už bylo vyzkoušeno. Všechny sítě používají referenční
implementaci, `lr: 0.01`, `batch_size: 32`, `seed: 42` a výchozí počty epoch
(500 pro regresi, 300 pro předpověď), mění se jen `hidden_dims`. Hodnoty jsou
na testovacích datech, tučně je nejlepší výsledek ve sloupci.

| `hidden_dims` | Parametrů (diabetes) | Sinusovka $R^2$ | Diabetes $R^2$ (500 epoch) | Diabetes $R^2$ (30 epoch) | Teploty MAE [°C] | Teploty RMSE [°C] |
|:---|---:|:---:|:---:|:---:|:---:|:---:|
| `[4]` | 49 | 0,924 | **0,386** | 0,381 | 1,772 | 2,231 |
| `[16]` | 193 | 0,965 | 0,269 | 0,374 | **1,729** | **2,189** |
| `[64]` | 769 | **0,978** | 0,269 | 0,373 | 1,737 | 2,213 |
| `[32, 16]` (výchozí) | 897 | 0,969 | 0,252 | 0,402 | 1,733 | 2,205 |
| `[64, 64]` | 4 929 | 0,977 | 0,194 | 0,391 | 1,757 | 2,236 |
| `[32, 32, 32]` | 2 497 | 0,971 | 0,192 | **0,403** | 1,751 | 2,221 |
| `[64, 32, 16]` | 3 329 | **0,978** | 0,235 | 0,394 | 1,763 | 2,245 |
| `[70, 70, 70, 70, 70, 70]` | 25 691 | 0,969 | 0,096 | 0,391 | 1,759 | 2,249 |

Pro srovnání, bez neuronové sítě:

| Naivní model | Výsledek |
|:---|:---|
| diabetes, lineární regrese (metoda nejmenších čtverců) | $R^2 = 0{,}385$ |
| teploty, persistence baseline | MAE 1,952 °C, RMSE 2,483 °C |

Co z tabulky plyne:

- **Hloubka sama nepomáhá.** Síť se šesti vrstvami a 25 000 parametry
  nevyhrála ani v jednom sloupci. Sinusovku proloží stejně jako výchozí síť
  a na teplotách je horší než jediná vrstva se 16 neurony.
- **U malého datasetu rozhoduje přeučení, ne kapacita.** Diabetes má 354
  trénovacích pacientů. Při 500 epochách je nejlepší nejmenší síť (49
  parametrů) a s rostoucí velikostí sítě $R^2$ klesá. Při 30 epochách se
  rozdíly mezi architekturami téměř ztratí. Počet epoch tu znamená víc než
  tvar sítě.
- **Náskok před lineární regresí je malý.** Nejlepší síť má u diabetu
  $R^2 = 0{,}403$, lineární regrese $0{,}385$. Ne každá úloha neuronovou síť
  potřebuje.
- **Na teplotách jsou všechny sítě podobné.** Liší se o setiny stupně a každá
  poráží persistence baseline o 9 až 11 %. Tvar sítě zde není úzké hrdlo.

**Ukažte, že to jde lépe.** Laťky k překonání:

| Úloha | Laťka |
|:---|:---|
| sinusovka | $R^2 > 0{,}978$ |
| diabetes | $R^2 > 0{,}403$ |
| teploty | MAE $< 1{,}729$ °C |

Měnit smíte vše v `config.yaml`: `hidden_dims`, `lr`, `epochs`, `batch_size`,
u předpovědi i `window_size`. Kód měnit nemusíte. Zapište si, které nastavení
jste zkusili a s jakým výsledkem, a zkuste zdůvodnit, proč vaše síť funguje
lépe.

> **Sekce `regression` je společná pro sinusovku i diabetes.** Nastavení,
> které pomůže jedné úloze, může druhé uškodit (krátké učení prospívá diabetu
> a škodí sinusovce). Hledejte buď kompromis, nebo každou laťku překonejte
> vlastním nastavením.

> **Pozor, i tohle je únik dat.** Vyberete-li z dvaceti pokusů ten
> s nejlepším výsledkem na **testovacích** datech, vybrali jste nastavení,
> které sedí právě těmto testovacím datům, a jeho skutečná chyba na nových
> datech bude horší. Je to totéž jako v Cvičení 06: testovací data smějí
> sloužit jen k závěrečnému vyhodnocení. Správně se hyperparametry ladí na
> zvláštní **validační** části oddělené z trénovacích dat. V tomto cvičení
> pro jednoduchost validační a testovací část splývají, proto berte překonání
> laťky jako cvičení v ladění sítě, ne jako důkaz, že je vaše síť obecně
> lepší.

---

## Lokální testování

Spusťte automatické testy příkazem:

```bash
pytest -v
```

| Třída testů | Co ověřuje |
|:---|:---|
| `TestMLP` | Tvar výstupu `(B, output_dim)`, počet a rozměry vrstev `nn.Linear`, `nn.Tanh` za každou skrytou vrstvou, **lineární výstupní vrstva** (výstup může ležet mimo $(-1, 1)$), registrace parametrů. |
| `TestTrainStep` | **Jádro:** krok vrací `float` rovný MSE **před** aktualizací, váhy po kroku jsou přesně $w - \eta\,\nabla w$ (odhalí i dvojitý `step`), opakované kroky snižují ztrátu, gradienty se mezi kroky nesčítají (`zero_grad`), nesouhlas tvarů `(B, 1)` a `(B,)` je chyba. |
| `TestTrainer` | Předvyplněné `fit`, `_evaluate` a `predict`: klíče a délky historie, pokles ztráty, model po `fit` i `predict` ve stavu `eval`, `predict` nestaví výpočetní graf; srozumitelné chyby pro špatný tvar cílů, prázdný loader a `_train_step`, který nevrací `float`. |
| `TestSaveLoad` | Uložení `state_dict` do `.pt`, načtení do nové instance se shodnými predikcemi, soubor je slovník tenzorů (ne celý model), načtení neexistujícího souboru je chyba. |
| `TestTabularDataset` | Hotový vzor: délka, tvary `(n_features,)` a `(1,)`, typ `float32`, hodnoty, dávky z `DataLoader`, různý počet vzorků je chyba. |
| `TestWindowedDataset` | **Okno nad signálem:** délka `len(series) - window_size`; na řadě, jejíž hodnoty se liší od indexů, je `x` minulých `window_size` hodnot a `y` hodnota následující (odhalí okénkování časové osy); první a poslední okno, tvary, typ, dávky, přímá iterace; 2D a příliš krátká řada jsou chyba. |
| `TestMetriky` | `mse`, `mae`, `r2` proti `sklearn.metrics`, známé hodnoty ($R^2 = 1$, $R^2 = 0$, záporné $R^2$), návratový typ `float`, kontrola tvarů. |
| `TestPersistenceBaseline` | Tvar, prvky jsou poslední známou hodnotou před cílem, číselný příklad z README, zarovnání s cíli `WindowedDataset`; odhalí **obě chyby o jedna** (vrácení cílů, posun o dva dny); výsledek `float64`, který nesdílí paměť se vstupem; 2D řada je chyba. |
| `TestResidualAnalyzer` | Předvyplněná diagnostika: rezidua, klíče testu normality, normální vs. šikmá rezidua, uložení grafů; chyby pro různou délku, NaN a méně než 3 hodnoty. |
| `TestDataAKonfigurace` | Tvary, standardizace z trénovací části a reprodukovatelnost dat, sinusovka, `chronological_split` bez míchání, `random_split` zachová páry, načtení konfigurace včetně sekcí `paths`, `sinusoid` a `diagnostics`, `ValueError` se jménem klíče pro špatné hodnoty ve všech sekcích a pro chybějící klíč. |
| `TestPipelineUtils` | Předvyplněný balíček `utils/`: kontrola učení pozná divergenci (nekonečná ztráta, NaN, prudký nárůst), přeučení a neklesající ztrátu; prahy se opravdu čtou ze sekce `diagnostics` v `config.yaml`; hlášky a predikce pipeline. |
| `TestUceniEndToEnd` | `MLP` s `Trainer` se na jednoduché závislosti naučí $R^2 > 0{,}9$. |
| `TestForecastNaTeplotach` | Načtení řady teplot (3650 hodnot), chronologické rozdělení 2920/730, zarovnání oken s baseline a MAE persistence na skutečných datech. **Přeskočí se, pokud soubor CSV chybí.** |

Testy tréninkového kroku a ukládání používají jako model samotnou vrstvu
`nn.Linear`. Jejich výsledek tak **nezávisí** na tom, zda máte hotovou třídu
`MLP`.

Dokud nejsou příslušné části hotové, testy, které je volají, se hlásí jako
**`xfail`** (očekávané selhání na `NotImplementedError`) a sada skončí
s návratovým kódem 0. Jakmile část doplníte, stejný test začne procházet
a ve výpisu se objeví jako `XPASS`. Hotová implementace nemá žádný `XFAIL`
ani `FAILED`: všechny testy jsou `PASSED` nebo `XPASS`. Nezapomeňte spustit
i celou pipeline:

```bash
python cviceni_11.py
```

---

## Doplňkové (papírové) příklady

Soubor `priklady_11.md` obsahuje příklady k ručnímu výpočtu ve stejné notaci
jako kód:

1. MSE, MAE, RMSE a $R^2$ ručně, předpověď průměrem ($R^2 = 0$), model
   horší než průměr ($R^2 < 0$) a vliv jedné odlehlé chyby na MSE a MAE,
2. past tvarů `(n,)` minus `(n, 1)`: co se ve skutečnosti spočítá a čemu se
   síť naučí,
3. okénkování krátké řady ručně, tvary dávek a chybné okénkování časové osy,
4. persistence baseline ručně, srovnání dvou modelů s baseline a obě chyby
   o jedna,
5. únik dat při náhodném rozdělení oken časové řady,
6. počty parametrů sítí a tři kroky gradientního sestupu na jediné váze se
   `zero_grad` a bez něj,
7. (volitelně) standardizace cíle a převod metrik zpět do původních jednotek,
8. příklady k procvičení a ukázka ověření výsledků funkcemi ze `src/`.

Čísla jsou volena tak, aby se dala spočítat na papíře. **Řešení nejsou
součástí repozitáře.** Výsledky si ověříte svými funkcemi ze `src/`.

---

## Odevzdání

Úloha se odevzdává ve **vaší kopii tohoto repozitáře** (vytvořené tlačítkem
*Use this template*). Po dokončení implementace proveďte:

```bash
git add src/model.py src/trainer.py src/datasets.py src/metrics.py
git commit -m "Implementace cvičení 11"
git push
```

> **Toto cvičení nemá automatické testy na GitHubu.** Na rozdíl od Cvičení
> 01–10 zde **záměrně chybí** workflow GitHub Actions (složka
> `.github/workflows/`). Instalace knihovny PyTorch při každém `push` by
> kontrolu neúměrně zpomalila. U commitu se proto nezobrazí zelená fajfka ani
> červený křížek a záložka **Actions** zůstane prázdná. Není to chyba.

> **Testy spouštějte lokálně.** Před odevzdáním spusťte ve virtuálním
> prostředí
>
> ```bash
> pytest -v
> python cviceni_11.py
> ```
>
> Hotová implementace nemá ve výpisu `pytest` žádný `XFAIL` ani `FAILED`
> (jen `PASSED` a `XPASS`) a pipeline nevypíše žádné `[NENI HOTOVO]` ani
> `[CHYBA IMPLEMENTACE]`.
> Lokální spuštění dává zpětnou vazbu za několik sekund, rychleji než
> automatická kontrola na GitHubu.

> **Soubory, které se neodevzdávají:** `src/__init__.py`,
> `src/diagnostics.py`, `dataio/` a `utils/` (celé balíčky), `cviceni_11.py`,
> `config.yaml`, `test_cviceni_11.py`, `requirements.txt`
> a `data/daily_min_temperatures.csv`. Jsou předvyplněny a nemají se měnit.
> `config.yaml` můžete pro experimenty upravovat lokálně, ale neodevzdávejte
> ho. Obsah složek `graphs/`, `logs/` a `models/` se generuje za běhu a do
> repozitáře nepatří.
