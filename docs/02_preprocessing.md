# NB2 — Preprocesamiento de señales

## Objetivo

La notebook `02_preprocessing.ipynb` implementa el pipeline que transforma las grabaciones polisomnográficas originales del dataset HMC en un conjunto de señales procesadas, alineadas y trazables, preparado para las etapas posteriores del proyecto.

La separación conceptual entre notebooks es:

- **NB1 — EDA:** analiza el dataset, valida supuestos y justifica decisiones metodológicas.
- **NB2 — Preprocessing:** aplica esas decisiones de forma reproducible sobre todos los sujetos.
- **NB3 — Feature engineering:** extrae features a partir de las señales procesadas.
- **Notebooks posteriores:** entrenan y evalúan modelos clásicos, gradient boosting y deep learning.

La NB2 intenta mantener un pipeline simple y general, evitando incorporar decisiones específicas de un modelo.

---

# 1. Datos de entrada

Para cada sujeto se utilizan:

- archivo EDF con la señal polisomnográfica;
- archivo de anotaciones de sleep staging;
- epochs de 30 segundos;
- cinco clases de sueño:
  - W;
  - N1;
  - N2;
  - N3;
  - REM.

Se conservan los 7 canales seleccionados durante el análisis exploratorio:

### EEG
- `F4-M1`
- `C4-M1`
- `O2-M1`
- `C3-M2`

### EOG
- `E1-M2`
- `E2-M2`

### EMG
- `Chin`

Los sujetos `SN032` y `SN033` se excluyen antes de generar el dataset final debido a problemas globales de calidad identificados en NB1.

---

# 2. Pipeline general

```text
EDF + anotaciones
        │
        ▼
1. Carga y selección de canales
        │
        ▼
2. Carga y validación de labels
        │
        ▼
3. Alineación por Recording onset
        │
        ▼
4. QC sobre señal cruda alineada
        │
        ▼
5. Filtrado continuo 0.5–35 Hz
        │
        ▼
6. Resampling 256 Hz → 100 Hz
        │
        ▼
7. Nueva extracción de epochs procesados
        │
        ▼
8. Cálculo de estadísticas de normalización
        │
        ▼
9. Eliminación únicamente de discard
        │
        ▼
10. Guardado del dataset y reportes
```

---

# 3. Carga y validación de labels

Los archivos de anotaciones se cargan con Pandas.

La notebook:

- limpia nombres de columnas y anotaciones;
- conserva únicamente filas correspondientes a `Sleep stage`;
- verifica que los sleep stages tengan una duración de 30 s;
- transforma W, N1, N2, N3 y REM a etiquetas enteras;
- conserva un `source_epoch_index` para mantener trazabilidad.

El criterio general utilizado es:

```text
Pandas → tablas, anotaciones, metadata y reportes
NumPy  → señales, epochs y operaciones numéricas
```

---

# 4. Alineación temporal mediante `Recording onset`

La NB2 no supone que todos los epochs son bloques perfectamente continuos de 30 s.

Cada anotación incluye un `Recording onset`, que representa el instante real en el que comienza ese sleep stage dentro de la grabación.

Para cada fila:

```text
Recording onset [s]
        ↓
multiplicar por frecuencia de muestreo
        ↓
muestra inicial de la señal
        ↓
extraer 30 segundos
```

Conceptualmente:

```text
start_sample = Recording onset × fs
end_sample   = start_sample + 30 × fs
```

Esto permite manejar correctamente:

- gaps entre anotaciones;
- epochs faltantes;
- irregularidades temporales;
- anotaciones fuera de la duración de la señal.

Para cada epoch se conservan:

- `source_epoch_index`;
- `recording_onset_s`.

De esta forma, cualquier epoch procesado puede rastrearse hasta su posición original.

---

# 5. Control de calidad por epoch

El QC se realiza sobre la **señal cruda ya alineada**, antes del filtrado y del resampling.

Para la decisión de calidad se utilizan los canales EEG.

## 5.1 Screening consistente con NB1

Un epoch se considera sospechoso si se cumple al menos una de estas condiciones:

```text
|amplitud EEG| > 500 µV
```

o

```text
flat >= 1 s
```

Estos criterios se utilizan como **screening**, no como criterio automático de descarte.

Una amplitud elevada por sí sola no elimina un epoch.

---

## 5.2 Estados de calidad

Cada epoch queda clasificado como:

### `ok`

No presenta anomalías detectadas por el screening.

### `review`

Presenta alguna anomalía, pero se conserva porque todavía puede contener información útil.

Ejemplos:

- amplitud EEG > 500 µV;
- flatline corto;
- flatline severo que no afecta a todos los EEG.

### `discard`

Se reserva para pérdida severa de señal:

```text
flatline continuo > 50 % del epoch
en todos los canales EEG seleccionados
```

La política es deliberadamente conservadora:

```text
amplitud elevada                  → review
flatline severo en algunos EEG    → review
flatline severo en todos los EEG  → discard
```

Así se evita eliminar de forma agresiva PSG clínico real.

---

## 5.3 Consistencia NB1 ↔ NB2

Se verifica explícitamente:

```text
suspicious_NB1 = review_NB2 ∪ discard_NB2
```

Es decir, un epoch marcado como sospechoso por la lógica de NB1 no puede terminar accidentalmente clasificado como `ok` en NB2.

---

## 5.4 Metadata de QC

Además del estado final, se registran métricas para facilitar la inspección posterior:

- amplitud máxima absoluta;
- duración máxima de flatline;
- fracción máxima del epoch afectada por flatline;
- número de canales EEG con flatline severo;
- canales afectados;
- motivo del screening;
- motivo de la acción final.

Estas variables se utilizan principalmente para **trazabilidad y análisis**, no para agregar nuevos criterios de descarte.

---

# 6. Exclusión de sujetos

NB1 identificó dos registros con problemas globales de adquisición:

```text
SN032
SN033
```

Ambos presentan deterioro persistente y multicanal, por lo que se excluyen a nivel sujeto antes de generar el dataset final.

La decisión se toma antes del entrenamiento de modelos.

Por lo tanto:

```text
151 sujetos originales
→ 149 sujetos disponibles para modelado
```

No se utiliza un porcentaje automático de epochs sospechosos para excluir otros sujetos.

---

# 7. Filtrado

Después del QC se aplica filtrado sobre la señal continua.

Configuración principal:

```text
band-pass: 0.5–35 Hz
```

El orden utilizado es:

```text
256 Hz
  ↓
filtrado 0.5–35 Hz
  ↓
100 Hz
```

Se filtra antes del resampling porque primero se define el rango frecuencial que se desea conservar y después se reduce la representación.

También se evita filtrar cada epoch por separado, reduciendo problemas de borde entre ventanas.

---

# 8. Notch de 50 Hz

No se utiliza un notch separado de 50 Hz.

Como el pipeline aplica un low-pass a 35 Hz:

```text
50 Hz queda fuera del rango conservado
```

por lo que un notch adicional sería redundante.

---

# 9. Resampling

La frecuencia de muestreo se reduce:

```text
256 Hz → 100 Hz
```

A 100 Hz, un epoch de 30 segundos contiene:

```text
3000 muestras por canal
```

Esto reduce memoria y costo computacional manteniendo el rango frecuencial seleccionado para el análisis principal, actualmente limitado a 35 Hz.

La elección de 100 Hz puede evaluarse posteriormente mediante una comparación con frecuencias superiores.

---

# 10. Nueva extracción de epochs procesados

Después del filtrado y del resampling se vuelven a extraer los epochs usando los mismos `Recording onset`.

Esto es necesario porque al cambiar la frecuencia de muestreo cambia la relación entre tiempo y número de muestra.

La notebook verifica que la señal cruda y la procesada mantengan:

- los mismos `source_epoch_index`;
- los mismos labels;
- los mismos `recording_onset_s`.

Si alguna de estas condiciones no se cumple, el procesamiento del sujeto se considera inconsistente.

---

# 11. Escala de la señal

La señal procesada se convierte de volts a:

```text
µV
```

y se guarda en esa escala.

No se realiza una normalización permanente dentro de NB2.

Esto permite que NB3 calcule features que dependen de la amplitud física de la señal, como:

- RMS;
- varianza;
- rango de amplitud;
- potencia absoluta.

---

# 12. Estadísticas para normalización posterior

Aunque la señal no se normaliza permanentemente, NB2 calcula por canal:

```text
media
desviación estándar
```

Preferentemente se utilizan epochs `ok`.

Si no existen epochs `ok`, pueden utilizarse los epochs conservables como fallback.

Estas estadísticas se guardan para que modelos posteriores puedan aplicar:

```text
x_norm = (x - media) / std
```

cuando sea necesario.

Así se evita duplicar almacenamiento guardando simultáneamente señal original y señal normalizada.

---

# 13. Selección final de epochs

La política final es:

```text
ok      → conservar
review  → conservar
discard → eliminar
```

Los epochs `review` siguen formando parte del dataset, pero quedan identificados mediante metadata.

Esto permite más adelante comparar, por ejemplo:

```text
entrenamiento con ok
vs.
entrenamiento con ok + review
```

sin volver a ejecutar el preprocessing.

---

# 14. Archivos generados

## 14.1 Dataset por sujeto

Para cada sujeto procesado:

```text
SNxxx.npz
```

Entre los campos principales se guardan:

```text
X
y
recording_onset_s
source_epoch_index
qc_status
qc_reason
review_mask
channel_names
channel_types
original_sfreq
processed_sfreq
channel_mean_uv
channel_std_uv
normalization_reference
subject_id
```

`X` contiene la señal procesada en µV y sin normalización permanente.

---

## 14.2 Reportes de calidad

Para cada sujeto:

```text
quality_reports/SNxxx_quality.csv
```

Estos archivos permiten inspeccionar la calidad epoch por epoch y recuperar el motivo de cada decisión.

---

## 14.3 Estadísticas de normalización

Para cada sujeto:

```text
normalization_stats/SNxxx_normalization_stats.csv
```

También se guardan las estadísticas dentro del `.npz`.

---

## 14.4 Resumen global

Al finalizar se genera:

```text
processing_summary.csv
```

que resume por sujeto:

- cantidad de labels;
- epochs alineados;
- epochs guardados;
- número de `ok`;
- número de `review`;
- número de `discard`;
- porcentaje de epochs sospechosos;
- porcentaje de review/discard;
- distribución final W/N1/N2/N3/REM.

---

## 14.5 Configuración reproducible

También se genera:

```text
preprocessing_config.json
```

Este archivo registra la configuración utilizada para producir el dataset:

- canales;
- duración de epoch;
- filtros;
- frecuencia de muestreo;
- criterios de QC;
- sujetos excluidos;
- unidad de señal;
- política de normalización.

Esto permite identificar con qué configuración se generaron los archivos procesados.

---

# 15. Análisis del impacto del QC por sleep stage

La notebook analiza también cuánto afecta el QC a cada clase.

Se estudian métricas como:

```text
n_original
n_suspicious
n_review
n_discard
n_kept
```

para:

```text
W
N1
N2
N3
REM
```

El objetivo es verificar que el preprocessing no esté eliminando de manera desproporcionada una clase particular.

Este análisis es especialmente importante para N1 por ser una clase menos representada y más difícil de clasificar.

---

# 16. Reutilización de resultados existentes

Procesar todos los EDF puede ser costoso.

La notebook puede verificar si un sujeto ya posee outputs completos:

```text
SNxxx.npz
SNxxx_quality.csv
SNxxx_normalization_stats.csv
```

y reutilizar el resultado anterior en vez de repetir todo el procesamiento.

Se utiliza:

```python
FORCE_REPROCESS = False
```

para omitir sujetos ya procesados.

Cuando cambia la metodología:

```python
FORCE_REPROCESS = True
```

permite regenerar todos los outputs.

---

# 17. Resultado conceptual

La salida de NB2 puede resumirse como:

```text
PSG original
     ↓
alineación temporal exacta
     ↓
QC conservador y trazable
     ↓
filtrado
     ↓
resampling
     ↓
epochs procesados en µV
     ↓
dataset reutilizable
    / \
   /   \
  ▼     ▼
NB3     Deep Learning
features     señal
```

La notebook busca dejar una representación intermedia **simple, reproducible, trazable y reutilizable**, sin incorporar todavía decisiones específicas del modelo.

---

# Referencias metodológicas utilizadas para QC

- Vallat, R. & Walker, M. P. (2021). *An open-source, high-performance tool for automated sleep staging*. eLife, 10:e70092. DOI: 10.7554/eLife.70092.
- Sommerauer et al. (2026). Trabajo basado en U-Sleep con criterios automáticos de calidad y flatline en PSG. *npj Digital Medicine*. DOI: 10.1038/s41746-026-02946-2.

Estas referencias respaldan una política conservadora frente a artefactos y el uso de pérdida sustancial de señal como criterio más fuerte que una excursión aislada de amplitud.
