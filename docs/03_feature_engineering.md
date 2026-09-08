# NB3 — Feature engineering

## Objetivo

La notebook `03_feature_engineering.ipynb` transforma los epochs procesados por NB2 en un dataset tabular de **features clásicas**, pensado como entrada para los modelos baseline de machine learning.

La separación conceptual entre las primeras notebooks queda:

- **NB1 — EDA:** analiza el dataset, valida supuestos y permite definir criterios de procesamiento.
- **NB2 — Preprocessing:** alinea, filtra, resamplea y aplica el control de calidad sobre las señales.
- **NB3 — Feature engineering:** resume cada epoch mediante variables temporales, espectrales y de complejidad.
- **Notebooks posteriores:** entrenan y evalúan los modelos utilizando splits agrupados por sujeto.

NB3 no entrena modelos ni realiza una selección definitiva de features. Su objetivo es generar una representación tabular reproducible, interpretable y trazable a partir de la señal procesada.

---

# 1. Datos de entrada

La notebook utiliza los archivos `.npz` generados por NB2 para cada sujeto.

En la ejecución actual se cargan:

```text
149 sujetos
135471 epochs
7 canales
3000 muestras por epoch y canal
100 Hz
30 s por epoch
```

La señal de entrada ya se encuentra:

- alineada con las anotaciones;
- filtrada entre `0.5–35 Hz`;
- resampleada a `100 Hz`;
- expresada en `µV`;
- libre de los epochs clasificados como `discard` en NB2.

Los epochs `review` se mantienen y continúan identificados mediante metadata.

Los canales son los mismos utilizados en NB2:

### EEG

- `EEG F4-M1`
- `EEG C4-M1`
- `EEG O2-M1`
- `EEG C3-M2`

### EOG

- `EOG E1-M2`
- `EOG E2-M2`

### EMG

- `EMG chin`

---

# 2. Pipeline general

```text
Outputs de NB2
      │
      ▼
1. Validación de cache y metadata
      │
      ▼
2. Carga de señales si es necesaria
      │
      ▼
3. Identificación del tipo de canal
      │
      ▼
4. Extracción de features por epoch y canal
      │
      ▼
5. Guardado recuperable por chunks
      │
      ▼
6. Consolidación del dataset tabular
      │
      ▼
7. Controles de integridad
      │
      ▼
8. EDA de las features
      │
      ▼
9. Catálogo y guardado final
```

Cada fila del dataset final representa un epoch de 30 segundos y conserva tanto las features calculadas como la metadata necesaria para rastrearlo hasta NB2.

---

# 3. Carga eficiente y reutilización de resultados

La extracción de features sobre todos los epochs requiere cargar una matriz de señales de aproximadamente `10.6 GB`.

Para evitar repetir este proceso innecesariamente, NB3 comprueba primero si existe un `features_full.parquet` compatible con la versión actual de los outputs de NB2.

El cache se reutiliza solamente si:

- contiene la metadata esperada;
- tiene la misma cantidad de epochs;
- contiene los mismos sujetos;
- es más reciente que los archivos de entrada de NB2;
- corresponde a la definición actual de features.

Si alguna de estas condiciones falla, se eliminan los resultados intermedios desactualizados y las features se vuelven a calcular.

De esta forma, un cambio en NB2 no deja mezclados resultados nuevos con chunks generados a partir de una versión anterior del preprocessing.

---

# 4. Trazabilidad

NB3 conserva para cada epoch:

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

Estas columnas no se consideran features del modelo.

`source_epoch_index` y `recording_onset_s` mantienen la relación con la secuencia original de anotaciones, mientras que `qc_status`, `qc_reason` y `review_mask` conservan la información de calidad generada en NB2.

Por lo tanto, las features pueden analizarse posteriormente según sujeto, etapa del sueño o estado de QC sin volver a cargar las señales originales.

---

# 5. Estrategia de extracción por canal

Las features se calculan **por separado para cada canal y cada epoch**.

La notebook identifica automáticamente el tipo de canal a partir de su nombre:

```text
EEG → features temporales + Hjorth + entropía + espectrales completas
EOG → features temporales + Hjorth + entropía + espectrales completas
EMG → features temporales + Hjorth + entropía + resumen espectral acotado
```

Las features espectrales del EMG se tratan de manera diferente porque NB2 limita todas las señales a `35 Hz`. Por lo tanto, en esta etapa no se intenta representar el espectro completo del EMG.

---

# 6. Features temporales

Para cada canal se calculan variables simples sobre las 3000 muestras del epoch:

```text
mean
std
rms
ptp
skew
kurt
zcr
```

Estas variables resumen distintos aspectos de la forma y amplitud de la señal:

- `mean`: valor medio;
- `std`: dispersión;
- `rms`: amplitud efectiva;
- `ptp`: rango peak-to-peak;
- `skew`: asimetría;
- `kurt`: curtosis;
- `zcr`: proporción de cruces por cero.

Las features se calculan directamente sobre la señal procesada en `µV`.

No se agrega una feature de varianza separada porque sería redundante con `hjorth_activity`, que corresponde a la varianza de la señal.

---

# 7. Parámetros de Hjorth

Para cada canal se calculan los tres parámetros de Hjorth:

```text
hjorth_activity
hjorth_mobility
hjorth_complexity
```

Conceptualmente:

- **activity:** representa la varianza de la señal;
- **mobility:** relaciona la variabilidad de la primera derivada con la señal original;
- **complexity:** compara la movilidad de la primera derivada con la movilidad de la señal.

Estas variables permiten resumir la dinámica temporal del epoch con un costo computacional bajo.

---

# 8. Entropía de permutación

También se calcula:

```text
perm_entropy
```

La implementación utiliza entropía de permutación de Bandt-Pompe con:

```text
order = 3
delay = 1
```

La feature resume el grado de regularidad o complejidad temporal de la señal a partir del orden relativo entre muestras consecutivas.

Se normaliza respecto de las seis permutaciones posibles, por lo que queda en una escala comparable entre canales y epochs.

---

# 9. Estimación espectral

Las features de frecuencia se obtienen a partir de la densidad espectral de potencia estimada con el método de Welch.

La configuración utiliza segmentos de aproximadamente:

```text
4 s
```

con:

```text
50 % de overlap
```

El análisis se restringe al mismo rango conservado durante NB2:

```text
0.5–35 Hz
```

Esto mantiene coherencia entre el preprocessing de la señal y las features extraídas posteriormente.

---

# 10. Features espectrales para EEG y EOG

Para los canales EEG y EOG se calcula la potencia en las siguientes bandas:

```text
delta  0.5–4 Hz
theta    4–8 Hz
alpha   8–13 Hz
sigma  12–15 Hz
beta   15–30 Hz
```

Para cada banda se obtiene:

```text
potencia absoluta
potencia relativa
```

La potencia relativa se calcula respecto de la potencia total entre `0.5–35 Hz`.

Además se guardan:

```text
total_power_0_5_35
spectral_entropy
sef95
```

donde `sef95` representa la frecuencia por debajo de la cual se acumula el 95 % de la potencia dentro del rango analizado.

También se calculan algunos cocientes espectrales:

```text
delta / beta
theta / alpha
delta / theta
sigma / beta
```

Estas relaciones permiten resumir cambios en el balance entre bandas sin depender únicamente de la potencia absoluta.

---

# 11. Features espectrales para EMG

El EMG utiliza un resumen espectral más acotado.

Se calculan:

```text
total_power_0_5_35
power_10_35
power_20_35
power_10_35_rel
power_20_35_rel
spectral_entropy
sef95
```

Esta decisión se debe a que la señal de NB2 ya fue limitada a `35 Hz`.

Por lo tanto, estas variables describen la información disponible dentro del rango conservado, pero no pretenden caracterizar todo el contenido frecuencial que puede presentar una señal EMG sin ese filtrado.

---

# 12. Cantidad total de features

La configuración actual genera:

```text
186 features por epoch
```

La distribución por tipo de canal es:

| Canal | Cantidad | Features por canal | Total |
|---|---:|---:|---:|
| EEG | 4 | 28 | 112 |
| EOG | 2 | 28 | 56 |
| EMG | 1 | 18 | 18 |
| **Total** | **7** |  | **186** |

Para EEG y EOG, cada canal incluye:

```text
7 temporales
3 Hjorth
1 entropía de permutación
17 espectrales
= 28 features
```

Para EMG:

```text
7 temporales
3 Hjorth
1 entropía de permutación
7 espectrales
= 18 features
```

---

# 13. Extracción recuperable por chunks

La extracción completa es una de las etapas más costosas de NB3.

Para evitar perder todo el procesamiento ante una interrupción, los epochs se procesan en chunks de:

```text
5000 epochs
```

Cada chunk se guarda como:

```text
features_XXXXXX_XXXXXX.parquet
```

Los chunks incluyen tanto las features como la metadata correspondiente.

Si la ejecución se interrumpe, la notebook identifica qué archivos ya existen y calcula solamente los faltantes.

En la ejecución actual, los `135471` epochs se dividen en `28` chunks.

---

# 14. Construcción y controles del dataset

Una vez disponibles todos los chunks, se concatenan y se ordenan mediante `epoch_global`.

La notebook verifica:

- que la cantidad final de filas coincida con los epochs de NB2;
- que los sujetos sean los mismos;
- que labels, metadata y features permanezcan alineados;
- que no existan infinitos;
- que no queden valores faltantes;
- que las features puedan representarse numéricamente;
- que no existan features constantes o prácticamente constantes.

Los valores no finitos se reemplazan por `NaN` y, si existen, se imputan con `0`.

En la ejecución actual:

```text
valores faltantes o no finitos: 0
features constantes o casi constantes: 0
```

Una vez consolidado el dataset tabular, la matriz original de señales se elimina de memoria.

---

# 15. EDA de las features

NB3 incluye una etapa de EDA breve antes del guardado final.

El objetivo no es seleccionar todavía las variables del modelo, sino comprobar que las features extraídas tengan comportamientos razonables y permitan distinguir patrones entre etapas del sueño.

Se realizan cuatro análisis principales.

## 15.1 Features candidatas por etapa

Se grafican ejemplos de distintas familias:

```text
delta relativa — EEG
sigma relativa — EEG
SEF95 — EEG
entropía de permutación — EEG
peak-to-peak — EOG
RMS — EMG
```

Los boxplots permiten observar cómo cambia su distribución entre:

```text
W
N1
N2
N3
REM
```

---

## 15.2 Perfil espectral promedio

Para un canal EEG principal se calcula la potencia relativa promedio de:

```text
delta
theta
alpha
sigma
beta
```

para cada etapa del sueño.

Este análisis sirve como control de plausibilidad de las features espectrales y permite comprobar que las distintas etapas presentan perfiles frecuenciales diferentes.

---

## 15.3 Correlación entre features

Se calcula la matriz de correlación absoluta entre las 186 variables.

Como diagnóstico se identifican pares con:

```text
|correlación| >= 0.95
```

En la ejecución actual se detectan `40` pares altamente correlacionados.

Esto se utiliza únicamente como diagnóstico. NB3 **no elimina automáticamente features correlacionadas**.

La decisión de seleccionar o reducir variables debe tomarse posteriormente dentro del pipeline de entrenamiento.

---

## 15.4 Separación univariada entre sleep stages

También se calcula un score exploratorio que compara:

```text
variabilidad entre clases
-------------------------
variabilidad dentro de las clases
```

El ranking permite identificar features que, consideradas individualmente, presentan mayor separación entre etapas.

En la ejecución actual aparecen entre las primeras posiciones variables como:

```text
EEG C3-M2__zcr
EEG C3-M2__perm_entropy
EEG F4-M1__perm_entropy
EEG C4-M1__zcr
```

Este ranking se utiliza **solo como EDA**.

Si posteriormente se realiza selección de features, debe hacerse utilizando exclusivamente los datos de entrenamiento dentro de cada split por sujeto. De lo contrario, se introduciría información del conjunto de validación o test.

---

# 16. Variabilidad entre sujetos

Para las features con mayor separación univariada se calcula también su promedio por:

```text
sujeto × sleep stage
```

Esto permite observar cuánto cambia una misma feature entre distintos registros aun dentro de la misma clase.

El análisis es relevante porque las señales fisiológicas presentan variabilidad inter-sujeto y refuerza la necesidad de evaluar los modelos separando sujetos completos entre train, validation y test.

---

# 17. Distribución de clases

NB3 conserva la distribución de sleep stages resultante de NB2.

En la ejecución actual:

| Stage | Epochs | Porcentaje |
|---|---:|---:|
| W | 23493 | 17.34 % |
| N1 | 14846 | 10.96 % |
| N2 | 49455 | 36.51 % |
| N3 | 26671 | 19.69 % |
| REM | 21006 | 15.51 % |

N2 es la clase más representada y N1 la menos frecuente.

La notebook registra esta distribución como parte del EDA, pero no aplica todavía técnicas de balanceo ni pesos de clase.

---

# 18. Catálogo de features

Además del dataset principal se genera:

```text
feature_catalog.csv
```

Para cada feature se registra:

```text
feature
channel
channel_type
family
variance
is_constant
```

Las familias utilizadas son:

```text
temporal
hjorth
entropy
spectral
```

Este catálogo facilita inspecciones posteriores y permite trabajar con grupos de variables sin depender de listas escritas manualmente.

---

# 19. Archivos generados

## 19.1 Dataset tabular

La salida principal se guarda como:

```text
features_full.parquet
```

Este archivo contiene:

```text
9 columnas de metadata
186 features
```

para un total de:

```text
195 columnas
```

La ejecución actual produce:

```text
135471 filas × 195 columnas
```

---

## 19.2 Dataset NumPy

También se genera:

```text
features_full.npz
```

que contiene principalmente:

```text
X_features
y
subjects
epoch_global
epoch_in_subject
source_epoch_index
recording_onset_s
qc_status
qc_reason
review_mask
feature_names
```

Este formato permite cargar directamente las matrices necesarias para las notebooks de modelado.

---

## 19.3 Catálogo

Se guarda:

```text
feature_catalog.csv
```

con la descripción estructural de las 186 features.

---

## 19.4 Figuras

La notebook genera figuras de apoyo para:

```text
candidate_features_boxplot.png
relative_band_power_heatmap.png
stage_distribution.png
top_univariate_features.png
subject_variability.png
```

Estas figuras se utilizan como controles exploratorios y para documentar el comportamiento de las variables extraídas.

---

# 20. Qué no hace NB3

Para mantener separadas las responsabilidades del pipeline, esta notebook **no realiza**:

- entrenamiento de modelos;
- separación train/validation/test;
- normalización o estandarización específica del modelo;
- balanceo de clases;
- selección definitiva de features;
- eliminación automática de features altamente correlacionadas;
- eliminación de epochs `review`;
- incorporación de contexto temporal entre epochs.

En particular, las 186 features describen cada epoch de manera independiente.

La información temporal (`source_epoch_index` y `recording_onset_s`) se conserva para poder incorporar contexto o dependencias entre epochs en etapas posteriores sin perder la secuencia original.

---

# 21. Prevención de leakage

La notebook puede analizar todas las features para fines de EDA, pero cualquier transformación que dependa de los datos y vaya a utilizarse durante el entrenamiento debe ajustarse posteriormente solo con el conjunto de train.

Esto incluye, por ejemplo:

```text
escalado
selección de features
reducción de dimensionalidad
imputación aprendida
```

Además, los splits deben realizarse a nivel sujeto.

La unidad correcta de separación es:

```text
subject
```

y no:

```text
epoch
```

De esta forma, epochs pertenecientes al mismo sujeto no quedan distribuidos entre train y test.

---

# 22. Resultado final

La ejecución actual produce:

```text
149 sujetos
135471 epochs
186 features por epoch
0 features constantes
9991 epochs marcados como review
```

La relación completa hasta esta etapa puede resumirse como:

```text
PSG original
     │
     ▼
NB1 — EDA
     │
     ▼
NB2 — preprocessing
     │
     ├── alineación
     ├── QC
     ├── filtrado
     └── resampling
     │
     ▼
epochs procesados en µV
     │
     ▼
NB3 — feature engineering
     │
     ├── temporales
     ├── Hjorth
     ├── entropía
     └── espectrales
     │
     ▼
186 features / epoch
     │
     ▼
dataset tabular
     │
     ▼
modelos clásicos
```

NB3 deja así una representación tabular **reproducible, interpretable y trazable**, manteniendo separadas la extracción de features y las decisiones que dependen del entrenamiento del modelo.
