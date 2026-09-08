# NB4 — Modelos clásicos y reducción de features

## Objetivo

La notebook `04_classical_baseline.ipynb` establece el primer baseline de clasificación utilizando las **186 features tabulares** generadas en NB3.

La idea principal es comparar modelos clásicos y estrategias simples de reducción de features bajo una evaluación consistente por sujeto.

Los modelos evaluados son:

- Dummy classifier;
- SGD con pérdida logística;
- Random Forest.

También se comparan tres configuraciones de entrada:

- todas las features;
- selección supervisada con ANOVA;
- selección supervisada con información mutua.

El criterio principal de comparación es **Macro-F1**, ya que el dataset presenta desbalance entre las cinco etapas del sueño.

---

# 1. Datos de entrada

La notebook utiliza:

```text
features_full.parquet
```

generado en NB3.

La ejecución actual contiene:

```text
135471 epochs
149 sujetos
186 features
```

Las columnas de metadata se mantienen separadas de las variables utilizadas para entrenar los modelos.

Entre ellas:

```text
epoch_global
epoch_in_subject
source_epoch_index
recording_onset_s
label
subject
qc_status
qc_reason
review_mask
```

Estas columnas se utilizan para trazabilidad y agrupamiento, pero no como features predictivas.

---

# 2. División train / test por sujeto

La separación se realiza con `GroupShuffleSplit`, utilizando el sujeto como grupo.

Configuración:

```text
test_size = 0.20
random_state = 42
```

La ejecución actual produce:

```text
Train: 108839 epochs — 119 sujetos
Test:   26632 epochs — 30 sujetos
```

No existe solapamiento de sujetos entre ambos conjuntos.

El split se guarda en:

```text
baseline_subject_split.json
```

Este archivo se reutiliza en las notebooks siguientes para asegurar que todos los modelos se comparen sobre exactamente el mismo holdout.

---

# 3. Por qué el split es por sujeto

Los epochs pertenecientes a un mismo registro comparten características fisiológicas, instrumentales y temporales.

Separar epochs aleatoriamente podría colocar información del mismo sujeto tanto en train como en test, produciendo una estimación demasiado optimista del desempeño.

Por eso, la unidad de separación utilizada en todo el pipeline de modelado es:

```text
subject
```

y no:

```text
epoch
```

---

# 4. Análisis exploratorio con Kruskal-Wallis

Antes de entrenar los modelos se realiza un análisis no paramétrico sobre el conjunto de entrenamiento.

Para reducir la dependencia entre miles de epochs del mismo sujeto, primero se calcula el promedio de cada feature por:

```text
sujeto × sleep stage
```

y luego se aplica Kruskal-Wallis entre las cinco etapas.

El resultado se utiliza como análisis exploratorio y se guarda en:

```text
kruskal_feature_ranking.csv
```

Entre las variables mejor posicionadas aparecen principalmente features espectrales y de complejidad de EEG, como:

```text
ratio_sigma_beta
beta_rel
zcr
beta_abs
perm_entropy
```

Este ranking no define directamente las features del modelo final.

---

# 5. Pipelines de modelado

Las transformaciones que utilizan información de los datos se incluyen dentro de pipelines de Scikit-learn para que se ajusten únicamente con el train correspondiente a cada fold.

## 5.1 SGD Logistic

El pipeline incluye:

```text
imputación por mediana
        ↓
eliminación de varianza casi nula
        ↓
selección de features, si corresponde
        ↓
StandardScaler
        ↓
SGDClassifier con log-loss
```

El clasificador utiliza:

```text
penalty = L2
class_weight = balanced
```

---

## 5.2 Random Forest

El pipeline incluye:

```text
imputación por mediana
        ↓
eliminación de varianza casi nula
        ↓
selección de features, si corresponde
        ↓
Random Forest
```

Configuración principal:

```text
100 árboles
max_depth = 20
min_samples_leaf = 5
max_features = sqrt
class_weight = balanced_subsample
```

Random Forest no requiere estandarización de las features.

---

# 6. Estrategias de selección de features

Se comparan dos métodos supervisados sobre un subconjunto de:

```text
50 features
```

## ANOVA

Utiliza:

```text
SelectKBest(f_classif)
```

## Información mutua

Utiliza:

```text
SelectKBest(mutual_info_classif)
```

Estas selecciones se realizan **dentro del pipeline y dentro de cada fold**.

Esto evita calcular la selección sobre todo el dataset antes de la validación y reduce el riesgo de leakage.

---

# 7. Cross-validation por sujeto

La comparación se realiza con:

```text
GroupKFold
3 folds
```

utilizando únicamente el conjunto de train.

Se evalúan:

```text
Dummy | All features
SGD | All features
SGD | ANOVA top 50
SGD | Mutual information top 50
Random Forest | All features
Random Forest | ANOVA top 50
```

Las métricas son:

```text
accuracy
balanced accuracy
Macro-F1
```

---

# 8. Resultados de cross-validation

Los resultados principales son:

| Configuración | Accuracy | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|
| Random Forest — todas las features | 0.7357 | 0.7043 | **0.7024** |
| SGD — todas las features | 0.7246 | 0.7057 | 0.6928 |
| Random Forest — ANOVA top 50 | 0.7062 | 0.6726 | 0.6685 |
| SGD — ANOVA top 50 | 0.7044 | 0.6673 | 0.6591 |
| SGD — información mutua top 50 | 0.6973 | 0.6614 | 0.6523 |
| Dummy | 0.3610 | 0.2000 | 0.1061 |

La mejor configuración es:

```text
Random Forest | All features
```

En esta primera comparación, reducir el espacio a 50 variables no mejora el desempeño respecto de utilizar las 186 features.

---

# 9. Selección del modelo final

El modelo final se selecciona utilizando únicamente los resultados de cross-validation sobre train.

El holdout permanece sin utilizar durante esta etapa.

La configuración seleccionada se vuelve a entrenar utilizando los:

```text
119 sujetos de train
108839 epochs
```

y recién después se evalúa sobre los 30 sujetos reservados para test.

---

# 10. Resultado sobre holdout

El modelo final obtiene:

```text
Accuracy          = 0.7343
Balanced accuracy = 0.7045
Macro-F1          = 0.7011
```

Métricas por clase:

| Etapa | Precision | Recall | F1 |
|---|---:|---:|---:|
| W | 0.7554 | 0.8277 | 0.7899 |
| N1 | 0.4129 | 0.3552 | 0.3819 |
| N2 | 0.7625 | 0.7563 | 0.7594 |
| N3 | 0.7964 | 0.7966 | 0.7965 |
| REM | 0.7696 | 0.7865 | 0.7780 |

N1 es claramente la etapa más difícil para este baseline.

---

# 11. Matriz de confusión

La notebook genera una matriz de confusión normalizada por clase:

```text
baseline_holdout_confusion_matrix.png
```

La normalización permite observar qué proporción de cada etapa real se asigna a las distintas clases, independientemente del número de ejemplos disponibles.

---

# 12. Importancia de features

Como el modelo seleccionado es Random Forest, se obtiene la importancia interna de las variables.

Entre las features con mayor importancia aparecen:

```text
EEG C3-M2__ratio_sigma_beta
EEG C4-M1__ratio_sigma_beta
EEG C4-M1__zcr
EEG C3-M2__zcr
EEG C3-M2__ratio_delta_beta
EMG chin__power_20_35
EEG F4-M1__ratio_sigma_beta
```

Esto sugiere que la discriminación se apoya en una combinación de:

- relaciones espectrales de EEG;
- medidas de complejidad y cruces por cero;
- información de actividad muscular.

La importancia se interpreta como una herramienta descriptiva y no como evidencia causal.

---

# 13. Archivos generados

## Split por sujeto

```text
baseline_subject_split.json
```

## Resultados de cross-validation

```text
baseline_feature_reduction_cv.csv
```

## Ranking exploratorio

```text
kruskal_feature_ranking.csv
```

## Modelo final

```text
baseline_best_model.joblib
```

## Predicciones de holdout

```text
baseline_holdout_predictions.parquet
```

Incluye metadata del epoch junto con:

```text
y_true
y_pred
```

## Métricas

```text
baseline_holdout_metrics.json
```

---

# 14. Controles de consistencia

La notebook verifica al finalizar:

- ausencia de solapamiento entre sujetos de train y test;
- exclusión de metadata de las features;
- existencia del split guardado;
- existencia de resultados de cross-validation;
- existencia del ranking Kruskal;
- guardado del modelo;
- guardado de predicciones;
- guardado de métricas;
- alineación entre predicciones y labels reales.

También invalida artefactos previos si detecta que `features_full.parquet` fue actualizado después de ellos.

---

# 15. Qué no hace NB4

NB4 no realiza:

- búsqueda exhaustiva de hiperparámetros;
- tuning utilizando el holdout;
- modelado de dependencias temporales entre epochs;
- entrenamiento sobre señal cruda;
- comparación final con boosting o deep learning.

Su objetivo es establecer una referencia clásica sólida y reproducible.

---

# 16. Resultado conceptual

```text
NB3
186 features / epoch
        │
        ▼
split por sujeto
        │
        ├───────────────┐
        ▼               ▼
     train            holdout
        │
        ▼
GroupKFold
        │
        ▼
modelos clásicos
+ selección de features
        │
        ▼
Random Forest
todas las features
        │
        ▼
evaluación final
        │
        ▼
Macro-F1 = 0.7011
```

NB4 deja definido tanto el **baseline clásico** como el **holdout común** que se reutiliza en las comparaciones posteriores.
