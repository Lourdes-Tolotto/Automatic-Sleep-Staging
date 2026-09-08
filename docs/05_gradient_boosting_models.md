# NB5 — Modelos de Gradient Boosting

## Objetivo

La notebook `05_gradient_boosting_models.ipynb` evalúa modelos de **gradient boosting** sobre las mismas 186 features generadas en NB3.

Se comparan:

- XGBoost;
- LightGBM;
- CatBoost.

La evaluación utiliza exactamente el mismo holdout por sujeto definido en NB4.

De esta forma, cualquier diferencia de desempeño entre el baseline clásico y los modelos de boosting puede atribuirse al modelo y no a un cambio en los sujetos utilizados para test.

El criterio principal de selección es:

```text
Macro-F1
```

---

# 1. Datos de entrada

La notebook utiliza:

```text
features_full.parquet
baseline_subject_split.json
```

El dataset contiene:

```text
135471 epochs
149 sujetos
186 features
```

El split heredado de NB4 queda:

```text
Train: 108839 epochs — 119 sujetos
Test:   26632 epochs — 30 sujetos
```

No existe solapamiento entre train y test.

---

# 2. Comparación directa con NB4

NB5 reutiliza directamente:

```text
baseline_subject_split.json
```

También carga los resultados del baseline cuando están disponibles:

```text
baseline_feature_reduction_cv.csv
baseline_holdout_metrics.json
```

Esto permite mostrar como referencia el desempeño obtenido anteriormente y comparar los nuevos modelos bajo las mismas condiciones.

---

# 3. Configuración general

Los tres modelos se entrenan con configuraciones fijas y relativamente simples.

No se realiza una búsqueda exhaustiva de hiperparámetros.

La intención de esta notebook es comparar familias de boosting en condiciones controladas antes de avanzar hacia un tuning más costoso.

Se utilizan hasta:

```text
4 threads por modelo
```

y:

```text
random_state = 42
```

---

# 4. XGBoost

Configuración principal:

```text
n_estimators = 250
learning_rate = 0.05
max_depth = 6
min_child_weight = 2
subsample = 0.85
colsample_bytree = 0.85
reg_alpha = 0.05
reg_lambda = 1.0
tree_method = hist
```

El objetivo es multiclase para las cinco etapas del sueño.

---

# 5. LightGBM

Configuración principal:

```text
n_estimators = 250
learning_rate = 0.05
num_leaves = 31
min_child_samples = 30
subsample = 0.85
colsample_bytree = 0.85
reg_alpha = 0.05
reg_lambda = 1.0
```

También se utiliza un objetivo multiclase.

---

# 6. CatBoost

Configuración principal:

```text
iterations = 250
learning_rate = 0.05
depth = 7
l2_leaf_reg = 3.0
```

El entrenamiento se ejecuta sobre CPU y sin generación de archivos auxiliares internos de CatBoost.

---

# 7. Manejo del desbalance

Los pesos se calculan únicamente a partir de las etiquetas disponibles en el conjunto de entrenamiento correspondiente.

Para cada fold se utiliza:

```text
compute_sample_weight(class_weight="balanced")
```

De esta forma, las clases menos frecuentes reciben mayor peso sin utilizar información del conjunto de validación.

La misma lógica se aplica al volver a entrenar el modelo seleccionado sobre todo el train.

---

# 8. Cross-validation por sujeto

La comparación se realiza con:

```text
GroupKFold
3 folds
```

sobre los 119 sujetos de train.

En cada fold se calculan:

```text
accuracy
balanced accuracy
Macro-F1
tiempo de entrenamiento
```

La notebook guarda los resultados de cada fold inmediatamente para permitir continuar una ejecución interrumpida.

---

# 9. Cross-validation recuperable

Los resultados individuales se almacenan en:

```text
boosting_fold_results.csv
```

Antes de entrenar un fold, la notebook verifica si esa combinación:

```text
modelo × fold
```

ya fue procesada.

Si existe, se reutiliza.

Esto evita repetir entrenamientos costosos cuando la ejecución se interrumpe.

---

# 10. Resultados de cross-validation

Los resultados medios son:

| Modelo | Accuracy | Balanced accuracy | Macro-F1 | Tiempo medio |
|---|---:|---:|---:|---:|
| LightGBM | **0.7427** | **0.7367** | **0.7250** | 27.6 s |
| XGBoost | 0.7387 | 0.7337 | 0.7210 | 44.2 s |
| CatBoost | 0.7258 | 0.7262 | 0.7106 | 153.6 s |
| SGD baseline | 0.7246 | 0.7057 | 0.6928 | 25.0 s |

Los tres modelos de boosting superan al SGD baseline en Macro-F1.

La mejor combinación entre desempeño y tiempo en esta configuración es LightGBM.

---

# 11. Selección del modelo final

El modelo final se selecciona exclusivamente mediante el:

```text
Macro-F1 medio de cross-validation
```

El holdout no participa en esta decisión.

El modelo seleccionado es:

```text
LightGBM
```

Luego se vuelve a entrenar utilizando los 119 sujetos de train y los pesos balanceados calculados sobre ese conjunto.

---

# 12. Resultado sobre holdout

LightGBM obtiene:

```text
Accuracy          = 0.7410
Balanced accuracy = 0.7330
Macro-F1          = 0.7219
```

Métricas por clase:

| Etapa | Precision | Recall | F1 |
|---|---:|---:|---:|
| W | 0.7989 | 0.7774 | 0.7880 |
| N1 | 0.3863 | 0.4906 | 0.4323 |
| N2 | 0.8170 | 0.7071 | 0.7581 |
| N3 | 0.7853 | 0.8508 | 0.8168 |
| REM | 0.7908 | 0.8388 | 0.8141 |

Respecto del baseline clásico, LightGBM mejora principalmente el equilibrio global entre clases y aumenta el recall de N1, aunque esta etapa continúa siendo la más difícil.

---

# 13. Matriz de confusión

Se genera una matriz de confusión normalizada:

```text
boosting_holdout_confusion_matrix.png
```

Esto permite analizar qué clases se confunden entre sí sin que el tamaño de cada clase domine visualmente el resultado.

---

# 14. Importancia de features

LightGBM expone una medida interna de importancia de variables.

Entre las primeras features aparecen:

```text
EEG O2-M1__ratio_theta_alpha
EMG chin__power_20_35
EEG C3-M2__ratio_sigma_beta
EMG chin__zcr
EMG chin__perm_entropy
EEG F4-M1__perm_entropy
EEG C3-M2__zcr
```

Al igual que en NB4, aparecen tanto variables espectrales de EEG como información del EMG y medidas de complejidad.

Estas importancias se utilizan con fines interpretativos y no como una selección definitiva de variables.

---

# 15. Comparación con el baseline clásico

En el mismo holdout:

```text
Random Forest
Macro-F1 = 0.7011

LightGBM
Macro-F1 = 0.7219
```

La mejora absoluta es de aproximadamente:

```text
+0.021 Macro-F1
```

Esto muestra que una familia de boosting puede aprovechar mejor las relaciones no lineales entre las features tabulares que el baseline clásico utilizado en NB4.

---

# 16. Archivos generados

## Resultados por fold

```text
boosting_fold_results.csv
```

## Resumen de cross-validation

```text
boosting_cv_results.csv
```

## Modelo final

```text
boosting_best_model.joblib
```

## Predicciones de holdout

```text
boosting_holdout_predictions.parquet
```

Incluye metadata y:

```text
y_true
y_pred
```

## Métricas de holdout

```text
boosting_holdout_metrics.json
```

---

# 17. Controles de consistencia

La notebook comprueba:

- ausencia de solapamiento entre sujetos de train y test;
- reutilización exacta del split de NB4;
- exclusión de metadata del espacio de features;
- existencia de resultados por fold;
- existencia del resumen de CV;
- guardado del modelo;
- guardado de predicciones;
- guardado de métricas;
- alineación de las predicciones.

Si NB3 o el split de NB4 fueron actualizados, los artefactos anteriores de NB5 se consideran desactualizados y se eliminan antes de volver a ejecutar.

---

# 18. Qué no hace NB5

NB5 no realiza:

- búsqueda exhaustiva de hiperparámetros;
- selección de modelo utilizando el test;
- modelado temporal entre epochs;
- entrenamiento sobre señales crudas;
- combinación o ensemble entre XGBoost, LightGBM y CatBoost.

El objetivo es establecer un baseline fuerte de boosting sobre features clásicas.

---

# 19. Resultado conceptual

```text
NB3
186 features / epoch
        │
        ▼
split de NB4
        │
        ▼
119 sujetos de train
        │
        ▼
GroupKFold
        │
        ├── XGBoost
        ├── LightGBM
        └── CatBoost
        │
        ▼
selección por Macro-F1
        │
        ▼
LightGBM
        │
        ▼
30 sujetos de test
        │
        ▼
Macro-F1 = 0.7219
```

NB5 mejora el baseline clásico manteniendo exactamente el mismo protocolo de evaluación.
