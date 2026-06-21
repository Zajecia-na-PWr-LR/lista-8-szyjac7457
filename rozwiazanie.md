# Lista 8 - Porównanie własnej pracy z rozwiązaniami konkursowymi

## 1. Wybór listy i konkursu

Do porównania wybrałem **Listę 3** - klasyfikacja binarna na zbiorze *Heart Disease Cleveland*
(przewidywanie kolumny `target`, 0 = zdrowy, 1 = chory).

Praca konkursowa:
**Playground Series - Season 6, Episode 2: "Predicting Heart Disease"**
https://www.kaggle.com/competitions/playground-series-s6e2

### Dlaczego ten konkurs jest adekwatny

- Ten sam typ problemu - klasyfikacja binarna na danych tabularycznych.
- Zbiór konkursowy wygenerowano syntetycznie z prawdziwego zbioru o chorobach serca, ma więc
  niemal identyczne cechy co mój Cleveland (Age, Sex, Chest pain type, BP, Cholesterol, Max HR,
  Thallium...). Target to `Heart Disease` (Presence/Absence). Porównanie etapów jest 1:1.
- Metryka konkursu to **ROC AUC** - dokładnie ta sama miara, którą sam liczyłem na końcu skryptu.

Wybrane notebooki (różne prace dla różnych obszarów), pobrane do tego folderu:

| Obszar | Notebook | Plik |
| --- | --- | --- |
| EDA | cosmicescape - *Comprehensive EDA + XGBoost* | `heart-disease-prediction-r2-tabular-playground.ipynb` |
| Preprocessing | yudainogata + masanakashima | `heart-disease-prediction-with-stacking-ensemble.ipynb`, `s6e2-heart-disease-top1-multi-seed.ipynb` |
| Optymalizacja hiperparametrów | yudainogata (parametry z Optuny) + cosmicescape | `heart-disease-prediction-with-stacking-ensemble.ipynb` |
| Uczenie modelu | masanakashima - *Top1 Multi-Seed* | `s6e2-heart-disease-top1-multi-seed.ipynb` |
| Ewaluacja | masanakashima + yudainogata | jw. |

Linki:
- https://www.kaggle.com/code/cosmicescape/heart-disease-s6e2-comprehensive-eda-xgboost
- https://www.kaggle.com/code/yudainogata/heart-disease-prediction-with-stacking-ensemble
- https://www.kaggle.com/code/masanakashima/s6e2-heart-disease-top1-multi-seed

Wszystkie trzy notebooki pochodzą bezpośrednio z konkursu S6E2. Fragmenty kodu poniżej są
przepisane z faktycznych komórek tych notebooków; mój kod pochodzi z `ML-lista3/zadania.py`.

## 2. Porównanie w 5 obszarach

### 2.1 Eksploracyjna analiza danych (EDA)

**Mój kod.** Brak EDA - skrypt od razu dzieli dane na X/y:

```python
df = pd.read_csv('Heart_disease_cleveland_new.csv')
X = df.drop(columns=['target'])
y = df['target']
```

**Praca konkursowa (cosmicescape, komórka 2).** Najpierw sprawdza balans klas:

```python
print(f"Target Distribution:\n{train['Heart Disease'].value_counts(normalize=True) * 100}")  # % każdej klasy
target_map = {'Absence': 0, 'Presence': 1}
train['Heart Disease'] = train['Heart Disease'].map(target_map)  # tekst -> 0/1
```

i bada zależność cecha-target wykresami (komórka 3):

```python
fig = px.histogram(train, x="Age", color="Heart Disease", marginal="box",  # rozkład wieku osobno dla chorych i zdrowych
                   title="Age Distribution vs Heart Disease Risk", barmode='overlay')
fig.show()
```

masanakashima robi z tego cały "Cardiologist's Dashboard" (4 panele, m.in. rozkład wieku wg
klasy z progiem Framingham = 55).

**Różnica / co mogłem lepiej.** Nie sprawdziłem nawet balansu klas - bez tego nie wiem, czy moje
accuracy to dobry wynik, czy efekt przewagi jednej klasy. EDA pokazałaby też zależności cech od
targetu i to, które kolumny są kategoryczne.

### 2.2 Preprocessing

**Mój kod.** `StandardScaler` na wszystkich kolumnach naraz, bez rozróżnienia typów:

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # skaluje WSZYSTKIE kolumny, też kategorie - to błąd
X_test_scaled = scaler.transform(X_test)
```

**Praca konkursowa (yudainogata, komórki 6-7).** Cechy kategoryczne i ciągłe traktowane osobno,
całość w `Pipeline`, kategorie kodowane TargetEncoderem, dopiero potem skalowanie:

```python
def feature_eng(X):
    X = X.copy()
    if 'id' in X.columns: X = X.drop(['id'], axis=1)
    X = X.drop(['FBS over 120'], axis=1)   # usuwa cechę nic nie wnoszącą
    return X

cat_cols = ['Chest pain type', 'EKG results', 'Slope of ST', 'Thallium']  # cechy kategoryczne
encoder_transformer = ColumnTransformer(
    transformers=[('target_enc', TargetEncoder(random_state=42), cat_cols)],  # koduj TYLKO kategorie
    remainder='passthrough')   # resztę (ciągłe) przepuść bez zmian

preprocessor = Pipeline([                          # kroki wykonywane po kolei
    ('feature_eng', FunctionTransformer(feature_eng)),
    ('encoder', encoder_transformer),
    ('scaler', StandardScaler())                   # skalowanie dopiero na końcu
])
```

masanakashima idzie dalej - target encoding **out-of-fold** (bez wycieku) + frequency encoding +
cechy eksperckie (komórki 6-7):

```python
df['rate_pressure_product'] = df['Max HR'] * df['BP'] / 1000   # obciążenie serca (nowa cecha)
theoretical_max = 220 - df['Age']
df['cardiac_reserve'] = (df['Max HR'] / theoretical_max).clip(0.5, 1.2)   # rezerwa serca
```

**Różnica / co mogłem lepiej.** Standaryzowanie zmiennej kategorycznej zakodowanej liczbą
("średni typ bólu") nie ma sensu. Trzeba skalować tylko cechy ciągłe, kategorie kodować
(target/one-hot), a najlepiej robić to **w `Pipeline`**, żeby przy walidacji nie było wycieku
danych. Inżynieria cech (jak `rate_pressure_product`) to często najmocniejsza dźwignia wyniku.

### 2.3 Optymalizacja hiperparametrów

**Mój kod.** Zero strojenia - domyślne parametry:

```python
log_reg = LogisticRegression(random_state=42)  # domyślne C=1.0, nic nie strojone
log_reg.fit(X_train_scaled, y_train)
```

**Praca konkursowa (yudainogata, komórka 8).** Parametry wyznaczone Optuną (widać "nieokrągłe"
wartości - efekt optymalizacji), osobno dla każdego modelu:

```python
xgb_params = {'n_estimators': 962, 'learning_rate': 0.0374, 'max_depth': 5,   # wartości znalezione przez Optunę
              'min_child_weight': 5, 'subsample': 0.973, 'colsample_bytree': 0.516,
              'gamma': 0.157}
cat_params = {'iterations': 838, 'learning_rate': 0.0998, 'depth': 3, 'l2_leaf_reg': 3, ...}
lgbm_params = {'n_estimators': 838, 'learning_rate': 0.0998, 'num_leaves': 151, ...}
```

cosmicescape również importuje `optuna` i `StratifiedKFold`, czyli stroi pod AUC z walidacji
krzyżowej, a nie na jednym splicie.

**Różnica / co mogłem lepiej.** Nawet zostając przy regresji logistycznej, mogłem zrobić
`GridSearchCV` po `C`, `penalty` i `solver`. Prace konkursowe stroją mocniejsze modele (boosting)
Optuną - i to one dowożą najwyższe AUC.

### 2.4 Uczenie modelu

**Mój kod.** Jeden model, jeden podział 80/20, jedno dopasowanie:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
log_reg.fit(X_train_scaled, y_train)
y_pred_log = log_reg.predict(X_test_scaled)
```

**Praca konkursowa (masanakashima, komórki 2, 8-10).** Multi-seed × 5-fold CV × trzy modele
boostingowe, uśrednianie OOF/predykcji, na końcu meta-model (stacking):

```python
SEEDS = [42, 2024, 2025, 1234, 5678]   # ten sam model na 5 ziarnach -> uśrednienie = stabilniejszy wynik
N_SPLITS = 5

def train_catboost_seed(X, y, X_test, seed):
    oof = np.zeros(len(X)); pred = np.zeros(len(X_test))
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)  # 5 foldów, proporcje klas zachowane
    for tr_i, va_i in skf.split(X, y):
        model = CatBoostClassifier(iterations=1500, learning_rate=0.05, depth=7,
                                   early_stopping_rounds=100, random_seed=seed)  # stop gdy przestaje się poprawiać
        model.fit(Pool(X.iloc[tr_i], y[tr_i]), eval_set=Pool(X.iloc[va_i], y[va_i]))
        oof[va_i] = model.predict_proba(X.iloc[va_i])[:, 1]   # predykcja na foldzie, którego model nie widział
        pred += model.predict_proba(X_test)[:, 1] / N_SPLITS  # średnia predykcji testu po foldach
    return oof, pred

# meta-model łączący CatBoost+XGB+LGBM
S_train = np.vstack([cb_oof_avg, xgb_oof_avg, lgb_oof_avg]).T   # wejście meta = predykcje 3 modeli
meta = LogisticRegression(max_iter=1000)
meta.fit(S_train, y)
```

yudainogata robi to prościej, gotowym `StackingClassifier` (komórka 9):

```python
stacking_model = Pipeline([
    ('preprocessor', preprocessor),
    ('stacking', StackingClassifier(
        estimators=[('xgb', XGBClassifier(**xgb_params)),         # modele bazowe...
                    ('cat', CatBoostClassifier(**cat_params)),
                    ('lgbm', LGBMClassifier(**lgbm_params))],
        final_estimator=LogisticRegression(), cv=5))   # ...meta-model je łączy, cv=5 chroni przed wyciekiem
])
```

**Różnica / co mogłem lepiej.** Regresja logistyczna to dobry, interpretowalny baseline, ale na
takich danych boosting bije ją na głowę. Minimalny krok: porównać kilka modeli pod CV; dalej -
ensemble (uśrednienie prawdopodobieństw) lub stacking. Multi-seed to tani sposób na stabilniejszy
wynik. Ciekawe, że nawet u nich meta-modelem jest... regresja logistyczna - czyli mój model żyje,
tyle że jako warstwa łącząca silniejsze modele.

### 2.5 Ewaluacja

**Mój kod.** Metryki na jednym splicie testowym (na plus: użyłem ROC/AUC = metryka konkursu):

```python
print(f"Accuracy logist.: {accuracy_score(y_test, y_pred_log):.4f}")
print(classification_report(y_test, y_pred_log))   # precision / recall / f1 per klasa
fpr, tpr, thresholds = roc_curve(y_test, y_prob_log)
roc_auc = auc(fpr, tpr)   # AUC, ale liczone tylko na jednym splicie
```

**Praca konkursowa (masanakashima, komórki 9-10).** AUC liczone **out-of-fold** i uśredniane po
ziarnach - stabilna, uczciwa estymacja:

```python
print(f"CB: {roc_auc_score(y, cb_oof):.5f} | XGB: {roc_auc_score(y, xgb_oof):.5f}")  # AUC na OOF = uczciwa ocena
stacked_auc = roc_auc_score(y, meta.predict_proba(S_train)[:, 1])   # AUC całego ensemble
```

yudainogata ocenia na wydzielonym, **stratyfikowanym** zbiorze walidacyjnym (komórki 10-11):

```python
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2,
                                                  random_state=42, stratify=y)  # stratify = te same proporcje klas w train/val
y_pred_proba = stacking_model.predict_proba(X_val)[:, 1]   # prawdopodobieństwa, nie etykiety
print(f'StackingClassifier ROC-AUC: {roc_auc_score(y_val, y_pred_proba):.4f}')
```

**Różnica / co mogłem lepiej.** Mój pojedynczy split na ~300 wierszach Cleveland daje wynik o dużej
wariancji - przy innym `random_state` AUC potrafi się zauważalnie zmienić. Powinienem liczyć
metryki przez CV i podawać średnią +/- odchylenie. Drobiazg, którego nie zrobiłem, a oni tak:
`stratify=y` przy podziale, żeby proporcje klas były takie same w train i test. Warto też dodać
krzywą precision-recall i przemyśleć próg decyzyjny - w medycynie false negative (przeoczony
chory) jest groźniejszy niż false positive.

## 3. Które czynniki najsilniej wpływają na wynik

1. **Wybór modelu** - przejście z regresji logistycznej na gradient boosting daje największy
   skok AUC na takich danych.
2. **Poprawny preprocessing i inżynieria cech** - rozdzielenie typów cech, sensowne kodowanie
   kategorii, cechy eksperckie; standaryzowanie kategorii (jak u mnie) wręcz szkodzi.
3. **Walidacja krzyżowa zamiast jednego splitu** - nie tyle podnosi wynik, co czyni go
   wiarygodnym i powtarzalnym.
4. **Strojenie (Optuna), multi-seed i stacking** - dokładają ostatnie ułamki AUC; w czołówce
   różnice to tysięczne części punktu.

## 4. Podsumowanie - co przeniósłbym do Listy 3

- Dodać realną EDA (balans klas, rozkłady, korelacje) przed modelowaniem.
- `Pipeline` + `ColumnTransformer`: kodowanie kategorii (target/one-hot), skalowanie tylko ciągłych.
- Podział z `stratify=y` i ocena 5-fold stratyfikowaną CV; AUC jako średnia z foldów.
- Dołożyć model drzewiasty i nastroić go (`GridSearchCV`, docelowo Optuna).
- Rozważyć ensemble/stacking oraz dobór progu decyzyjnego pod koszt błędów w kontekście medycznym.

Mój baseline jest poprawny w zakresie, w jakim go napisałem (skalowanie po splicie, użycie AUC),
ale to punkt startowy - prace konkursowe pokazują, że największe zyski leżą w preprocessingu,
silniejszym modelu i wiarygodnej walidacji.
