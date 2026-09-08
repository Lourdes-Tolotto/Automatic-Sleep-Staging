# NB7 — Comparación de modelos y análisis de errores

## Objetivo

La notebook `07_model_comparison_and_error_analysis.ipynb` reúne los resultados finales de los tres enfoques evaluados hasta el momento:

- baseline clásico de NB4;
- gradient boosting de NB5;
- 1D-CNN de NB6.

La comparación utiliza las predicciones guardadas por cada notebook sobre **exactamente el mismo holdout**.

El objetivo no es volver a entrenar modelos, sino analizar:

- desempeño global;
- desempeño por etapa del sueño;
- matrices de confusión;
- comportamiento según estado de QC;
- variabilidad entre sujetos.

---

# 1. Modelos comparados

Las configuraciones finales son:

```text
Classical baseline
→ Random Forest | All features

Gradient boosting
→ LightGBM

Deep learning
→ 1D-CNN
```

Los tres modelos se evalúan sobre:

```text
26632 epochs
30 sujetos
```

---

# 2. Datos de entrada

La notebook carga las métricas:

```text
baseline_holdout_metrics.json
boosting_holdout_metrics.json
dl_cnn_holdout_metrics.json
```

y las predicciones:

```text
baseline_holdout_predictions.parquet
boosting_holdout_predictions.parquet
dl_cnn_holdout_predictions.parquet
```

No se accede nuevamente a los datos de entrenamiento.

---

# 3. Validación de alineación

Antes de comparar resultados se verifica que las tres tablas tengan exactamente los mismos epochs.

Las claves utilizadas son:

```text
epoch_global
subject
source_epoch_index
recording_onset_s
```

También se comprueba que coincidan:

```text
y_true
qc_status
```

Si existe alguna diferencia, la notebook detiene la comparación.

Esto evita comparar métricas obtenidas sobre subconjuntos de test distintos.

---

# 4. Comparación global

Los resultados finales son:

| Familia | Modelo | Accuracy | Balanced accuracy | Macro-F1 |
|---|---|---:|---:|---:|
| Deep learning | 1D-CNN | **0.7437** | **0.7541** | **0.7359** |
| Gradient boosting | LightGBM | 0.7410 | 0.7330 | 0.7219 |
| Classical baseline | Random Forest | 0.7343 | 0.7045 | 0.7011 |

El mejor resultado global corresponde a:

```text
1D-CNN
```

La diferencia en accuracy entre los modelos es relativamente pequeña, pero se vuelve más clara al observar balanced accuracy y Macro-F1.

Esto indica que el principal beneficio del modelo de deep learning aparece en el equilibrio entre clases y no solamente en la cantidad total de predicciones correctas.

---

# 5. Métricas por etapa del sueño

La notebook calcula para cada modelo:

```text
precision
recall
F1
support
```

sobre:

```text
W
N1
N2
N3
REM
```

---

# 6. Wake

F1 por modelo:

```text
Random Forest → 0.7899
LightGBM      → 0.7880
1D-CNN        → 0.7868
```

Los tres enfoques presentan resultados muy similares para W.

La CNN tiene mayor precision, pero menor recall.

---

# 7. N1

F1 por modelo:

```text
Random Forest → 0.3819
LightGBM      → 0.4323
1D-CNN        → 0.4718
```

N1 continúa siendo la clase más difícil.

La CNN aumenta considerablemente el recall:

```text
Random Forest → 0.3552
LightGBM      → 0.4906
1D-CNN        → 0.6405
```

Sin embargo, su precision en N1 es solamente:

```text
0.3734
```

Por lo tanto, parte de la mejora se obtiene prediciendo N1 de forma más frecuente, lo que también genera falsos positivos.

---

# 8. N2

F1:

```text
Random Forest → 0.7594
LightGBM      → 0.7581
1D-CNN        → 0.7540
```

Los tres modelos presentan un desempeño muy parecido.

N2 es la clase más frecuente del dataset, pero no muestra una ventaja clara para el modelo más complejo.

---

# 9. N3

F1:

```text
Random Forest → 0.7965
LightGBM      → 0.8168
1D-CNN        → 0.8379
```

La CNN obtiene el mejor resultado.

Su recall alcanza:

```text
0.8953
```

lo que indica una buena sensibilidad para sueño profundo.

---

# 10. REM

F1:

```text
Random Forest → 0.7780
LightGBM      → 0.8141
1D-CNN        → 0.8291
```

Tanto boosting como deep learning mejoran claramente sobre el baseline clásico.

La CNN vuelve a obtener el mejor resultado.

---

# 11. Matrices de confusión

Se genera una matriz de confusión normalizada para cada modelo:

```text
nb7_baseline_confusion_matrix.png
nb7_boosting_confusion_matrix.png
nb7_cnn_confusion_matrix.png
```

El objetivo es comparar no solo cuántos errores comete cada modelo, sino **entre qué etapas se producen esas confusiones**.

Esto es especialmente importante en sleep staging, donde ciertas transiciones fisiológicas son difíciles incluso para modelos con buen desempeño global.

---

# 12. Desempeño según estado de QC

Como NB2 conserva los epochs `review`, NB7 puede medir el rendimiento por separado sobre:

```text
ok
review
```

## Epochs `ok`

Resultados en Macro-F1:

```text
Random Forest → 0.7061
LightGBM      → 0.7283
1D-CNN        → 0.7418
```

## Epochs `review`

Resultados en Macro-F1:

```text
Random Forest → 0.4831
LightGBM      → 0.5219
1D-CNN        → 0.5636
```

El desempeño cae de forma clara en los epochs marcados para revisión.

Esto confirma que las banderas de QC contienen información relevante sobre la dificultad de clasificación.

---

# 13. Interpretación del análisis de QC

El conjunto de test contiene:

```text
24209 epochs ok
2423 epochs review
```

Los tres modelos empeoran sobre `review`, aunque el orden relativo se mantiene aproximadamente:

```text
1D-CNN
>
LightGBM
>
Random Forest
```

Esta comparación también respalda la decisión tomada en NB2 de **conservar los review pero mantenerlos identificados**, en lugar de descartarlos automáticamente.

Permite cuantificar directamente qué impacto tienen sobre los modelos.

---

# 14. Variabilidad entre sujetos

La notebook calcula Macro-F1 por sujeto para cada modelo.

Resumen:

| Modelo | Media | Std | Mediana | Mínimo | Máximo |
|---|---:|---:|---:|---:|---:|
| 1D-CNN | **0.7196** | 0.0920 | 0.7370 | 0.4395 | 0.8367 |
| LightGBM | 0.7103 | 0.1031 | 0.7345 | 0.4152 | **0.8399** |
| Random Forest | 0.6784 | 0.1062 | 0.7212 | 0.4346 | 0.8183 |

La media por sujeto vuelve a favorecer a la CNN.

Sin embargo, la amplitud entre los mejores y peores registros muestra que todavía existe una variabilidad inter-sujeto importante.

---

# 15. Por qué analizar por sujeto

Una única métrica calculada sobre todos los epochs puede quedar dominada por sujetos con noches más largas o con mayor cantidad de ejemplos.

Calcular Macro-F1 individualmente para cada sujeto permite observar si el modelo funciona de forma consistente entre registros.

Este análisis es especialmente importante para evaluar generalización a sujetos no vistos.

---

# 16. Archivos generados

## Comparación global

```text
model_comparison_holdout.csv
```

## Métricas por etapa

```text
model_comparison_per_stage.csv
```

## Comparación según QC

```text
model_comparison_qc.csv
```

## Resultados por sujeto

```text
model_comparison_per_subject.csv
```

## Figuras

```text
nb7_baseline_confusion_matrix.png
nb7_boosting_confusion_matrix.png
nb7_cnn_confusion_matrix.png
nb7_subject_f1_distribution.png
```

---

# 17. Controles de consistencia

La notebook verifica:

- mismo número de epochs para NB4, NB5 y NB6;
- misma identidad y orden de los epochs;
- mismos sujetos;
- mismos labels;
- mismo estado de QC;
- guardado de la comparación global;
- guardado de métricas por etapa;
- guardado del análisis de QC;
- guardado del análisis por sujeto.

Por lo tanto, la comparación se realiza sobre una base común y directamente comparable.

---

# 18. Resultado principal

El orden final por Macro-F1 es:

```text
1D-CNN       → 0.7359
LightGBM     → 0.7219
Random Forest → 0.7011
```

La mejora desde el baseline clásico hasta la CNN es aproximadamente:

```text
+0.035 Macro-F1
```

El deep learning muestra sus ventajas principalmente en:

- N1;
- N3;
- REM;
- balanced accuracy;
- desempeño promedio entre sujetos.

---

# 19. Limitaciones que siguen abiertas

A pesar de ser el mejor modelo, la 1D-CNN todavía presenta limitaciones importantes:

- N1 sigue teniendo un F1 inferior a `0.50`;
- existe variabilidad considerable entre sujetos;
- los epochs `review` son notablemente más difíciles;
- cada epoch se clasifica de forma independiente;
- todavía no se utiliza la estructura temporal de la secuencia de sueño.

Por lo tanto, el resultado no debe interpretarse como el cierre del modelado.

---

# 20. Próximo paso

La siguiente mejora propuesta es incorporar **contexto temporal entre epochs** sobre la representación de deep learning.

La motivación es que el sleep staging no es una clasificación de observaciones independientes.

La secuencia:

```text
... → N2 → N2 → N3 → N3 → N2 → REM → ...
```

contiene información sobre las transiciones posibles entre etapas.

Un modelo temporal podría utilizar esta información para resolver epochs ambiguos, especialmente en clases como N1.

---

# 21. Resultado conceptual

```text
                    NB4
              Random Forest
                    │
                    │
NB3 features ───────┼──────── NB5
                    │        LightGBM
                    │
                    ▼
              mismo holdout
                    ▲
                    │
NB2 señales ──── NB6
                 1D-CNN
                    │
                    ▼
                   NB7
          comparación y errores
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
      por clase    QC     por sujeto
                    │
                    ▼
              mejor modelo
                 1D-CNN
                    │
                    ▼
          contexto temporal
```

NB7 consolida la primera comparación completa del proyecto y deja identificado el siguiente paso metodológico: pasar de una clasificación independiente por epoch a un modelo que aproveche la estructura temporal del sueño.
