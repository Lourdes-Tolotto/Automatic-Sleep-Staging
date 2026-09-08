# NB6 — Baseline de Deep Learning con 1D-CNN

## Objetivo

La notebook `06_deep_learning_cnn_baseline.ipynb` introduce el primer modelo de **deep learning** del proyecto.

A diferencia de NB4 y NB5, que trabajan sobre las features tabulares de NB3, esta notebook utiliza directamente las señales multicanal procesadas en NB2.

Cada entrada al modelo representa:

```text
7 canales × 3000 muestras
```

correspondientes a un epoch de 30 segundos a `100 Hz`.

El objetivo es establecer un baseline simple de deep learning antes de incorporar contexto temporal entre epochs.

---

# 1. Datos de entrada

La notebook utiliza los archivos:

```text
SNxxx.npz
```

generados en NB2.

La ejecución actual contiene:

```text
149 sujetos
135471 epochs
7 canales
3000 muestras por canal
100 Hz
señal en µV
```

Los siete canales son:

```text
EEG F4-M1
EEG C4-M1
EEG O2-M1
EEG C3-M2
EOG E1-M2
EOG E2-M2
EMG chin
```

Los epochs `discard` ya fueron eliminados en NB2.

Los `review` se mantienen y conservan su metadata.

---

# 2. Cache de señales

Para evitar abrir 149 archivos `.npz` repetidamente durante el entrenamiento, la notebook construye un cache continuo:

```text
dl_signals.npy
dl_metadata.parquet
```

`dl_signals.npy` contiene todas las señales en una única matriz:

```text
[n_epochs, 7, 3000]
```

y se abre mediante `mmap_mode="r"` para no cargar los aproximadamente 10 GB completos en RAM.

La metadata conserva:

```text
epoch_global
subject
label
source_epoch_index
recording_onset_s
qc_status
qc_reason
review_mask
```

Si los archivos de NB2 son más recientes que el cache, este se reconstruye automáticamente.

---

# 3. División train / validation / test por sujeto

El conjunto de test es exactamente el mismo definido en NB4:

```text
30 sujetos
26632 epochs
```

Los 119 sujetos restantes se vuelven a dividir por sujeto para obtener train y validation.

La ejecución actual queda:

```text
Train:       87307 epochs — 95 sujetos
Validation:  21532 epochs — 24 sujetos
Test:        26632 epochs — 30 sujetos
```

No existe solapamiento de sujetos entre ninguno de los tres conjuntos.

La partición específica de deep learning se guarda en:

```text
dl_subject_split.json
```

---

# 4. Distribución de clases

El conjunto de train contiene:

| Etapa | Epochs |
|---|---:|
| W | 15703 |
| N1 | 9609 |
| N2 | 31712 |
| N3 | 16740 |
| REM | 13543 |

N2 sigue siendo la clase dominante y N1 la menos representada.

Esta distribución se tiene en cuenta durante el entrenamiento mediante pesos de clase.

---

# 5. Normalización sin leakage

Las estadísticas de normalización se calculan **únicamente con los sujetos de train**.

Para cada canal se obtiene:

```text
media
desviación estándar
```

y luego cada epoch se normaliza como:

```text
x_norm = (x - mean_train) / std_train
```

Las estadísticas se reutilizan para train, validation y test.

Nunca se calculan utilizando validation o test.

Se guardan en:

```text
dl_channel_normalization.json
```

La desviación estándar obtenida en train se encuentra aproximadamente entre:

```text
31 y 38 µV
```

según el canal.

---

# 6. Dataset y DataLoaders

La normalización se aplica al cargar cada muestra desde el cache, sin crear una segunda copia normalizada de todas las señales.

Cada elemento del dataset devuelve:

```text
x → [7, 3000]
y → etapa del sueño
```

Configuración:

```text
batch_size = 128
```

La ejecución actual genera:

```text
683 batches de train
169 batches de validation
209 batches de test
```

---

# 7. Arquitectura 1D-CNN

Se utiliza una CNN temporal multicanal relativamente compacta.

La arquitectura es:

```text
Entrada: 7 × 3000
        │
        ▼
Conv1D 7 → 32, kernel 7
BatchNorm
ReLU
MaxPool ×4
        │
        ▼
Conv1D 32 → 64, kernel 7
BatchNorm
ReLU
MaxPool ×4
        │
        ▼
Conv1D 64 → 128, kernel 5
BatchNorm
ReLU
        │
        ▼
Adaptive Average Pooling
        │
        ▼
Dropout 0.30
        │
        ▼
Linear 128 → 5
```

El modelo contiene:

```text
58181 parámetros entrenables
```

La arquitectura se mantiene deliberadamente simple para utilizarla como baseline.

---

# 8. Manejo del desbalance

Se utilizan pesos de clase inversamente proporcionales a la frecuencia en train.

Los pesos obtenidos son aproximadamente:

| Etapa | Peso |
|---|---:|
| W | 1.1120 |
| N1 | 1.8172 |
| N2 | 0.5506 |
| N3 | 1.0431 |
| REM | 1.2893 |

N1 recibe el mayor peso y N2 el menor.

Los pesos se incorporan en:

```text
CrossEntropyLoss
```

---

# 9. Optimización

El optimizador es:

```text
Adam
learning_rate = 1e-3
weight_decay = 1e-4
```

La configuración de entrenamiento utiliza:

```text
máximo 25 epochs
patience = 5
```

La métrica utilizada para conservar el mejor checkpoint es:

```text
validation Macro-F1
```

---

# 10. Early stopping

Después de cada epoch se evalúa el modelo sobre validation.

Si el Macro-F1 mejora, el checkpoint se guarda.

Si no mejora durante cinco epochs consecutivos, el entrenamiento se detiene.

En la ejecución actual:

```text
mejor epoch aproximado: 8
mejor validation Macro-F1 = 0.7403
early stopping: epoch 13
```

El conjunto de test no se utiliza durante el entrenamiento ni para decidir cuándo detenerlo.

---

# 11. Curvas de entrenamiento

La notebook guarda la evolución de:

```text
train loss
validation loss
validation Macro-F1
```

en:

```text
dl_cnn_training_history.csv
```

También genera figuras para inspeccionar el comportamiento del entrenamiento y detectar posibles problemas de convergencia o sobreajuste.

---

# 12. Evaluación final sobre holdout

Después del entrenamiento se recupera el checkpoint con mejor Macro-F1 de validation.

El test se evalúa una sola vez.

El resultado es:

```text
Accuracy          = 0.7437
Balanced accuracy = 0.7541
Macro-F1          = 0.7359
```

Métricas por clase:

| Etapa | Precision | Recall | F1 |
|---|---:|---:|---:|
| W | 0.8787 | 0.7122 | 0.7868 |
| N1 | 0.3734 | 0.6405 | 0.4718 |
| N2 | 0.8624 | 0.6698 | 0.7540 |
| N3 | 0.7874 | 0.8953 | 0.8379 |
| REM | 0.8065 | 0.8529 | 0.8291 |

El uso de pesos de clase aumenta notablemente el recall de N1 respecto de los modelos clásicos, aunque su precisión sigue siendo baja.

---

# 13. Comparación preliminar

Sobre exactamente el mismo holdout:

```text
Random Forest
Macro-F1 = 0.7011

LightGBM
Macro-F1 = 0.7219

1D-CNN
Macro-F1 = 0.7359
```

La CNN obtiene el mejor desempeño global de las tres familias evaluadas hasta esta etapa.

Esto es relevante porque el modelo aprende directamente desde la señal y no depende de la extracción manual de las 186 features de NB3.

---

# 14. Archivos generados

## Cache de señal

```text
dl_signals.npy
dl_metadata.parquet
```

## Split

```text
dl_subject_split.json
```

## Normalización

```text
dl_channel_normalization.json
```

## Modelo

```text
dl_cnn_baseline.pt
```

## Historial de entrenamiento

```text
dl_cnn_training_history.csv
```

## Predicciones

```text
dl_cnn_holdout_predictions.parquet
```

## Métricas

```text
dl_cnn_holdout_metrics.json
```

Las predicciones mantienen:

```text
epoch_global
subject
source_epoch_index
recording_onset_s
qc_status
qc_reason
review_mask
y_true
y_pred
```

---

# 15. Controles de consistencia

La notebook verifica:

- siete canales;
- 3000 muestras por epoch;
- frecuencia de 100 Hz;
- señal almacenada en µV;
- ausencia de solapamiento entre train, validation y test;
- uso del mismo test de NB4;
- cálculo de normalización únicamente sobre train;
- existencia del checkpoint;
- guardado del historial;
- guardado de predicciones;
- guardado de métricas.

---

# 16. Qué no hace NB6

Este baseline todavía clasifica cada epoch de forma independiente.

No incorpora:

- epochs anteriores;
- epochs posteriores;
- secuencias completas;
- GRU o LSTM;
- Transformers temporales;
- postprocesamiento secuencial.

Por lo tanto, el modelo utiliza la estructura temporal **dentro de los 30 segundos del epoch**, pero no la transición entre etapas consecutivas.

---

# 17. Próximo paso

La mejora metodológica más natural es incorporar contexto temporal.

Conceptualmente:

```text
epoch t-2
epoch t-1
epoch t
epoch t+1
epoch t+2
     │
     ▼
representación temporal
     │
     ▼
sleep stage de t
```

Esto permitiría aprovechar que las etapas del sueño presentan transiciones estructuradas y no son observaciones independientes.

---

# 18. Resultado conceptual

```text
NB2
señales procesadas
7 × 3000
        │
        ▼
split por sujeto
        │
        ▼
normalización desde train
        │
        ▼
1D-CNN
        │
        ▼
representación aprendida
directamente desde señal
        │
        ▼
5 clases
        │
        ▼
holdout común
        │
        ▼
Macro-F1 = 0.7359
```

NB6 establece el primer baseline de deep learning y deja preparada la base para incorporar contexto temporal en las siguientes iteraciones.
