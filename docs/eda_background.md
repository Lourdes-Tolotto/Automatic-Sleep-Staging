```markdown
# Contexto técnico para el análisis exploratorio

Este documento reúne conceptos técnicos utilizados durante el análisis exploratorio del dataset de sleep staging. El objetivo es mantener las notebooks enfocadas en el procesamiento y los resultados, dejando aquí el contexto adicional sobre MNE, archivos EDF y análisis espectral de señales EEG.

---

# 1. MNE-Python y archivos EDF

## 1.1 MNE-Python

**MNE-Python** es una librería de Python de código abierto orientada al análisis y procesamiento de señales neurofisiológicas y biomédicas, como EEG, MEG, EOG, ECG y EMG.

En este proyecto se utiliza principalmente para:

- leer las grabaciones almacenadas en formato EDF;
- acceder a las señales y sus metadatos;
- seleccionar y tipificar canales;
- realizar operaciones de procesamiento como resampling y filtrado;
- facilitar la inspección temporal y espectral de las señales.

---

## 1.2 Archivos EDF

**EDF** (*European Data Format*) es un formato utilizado para almacenar señales biomédicas multicanal.

Un archivo EDF puede contener:

- señales registradas a lo largo del tiempo;
- nombres de canales;
- frecuencia de muestreo;
- información sobre unidades físicas;
- duración de la grabación;
- información temporal;
- anotaciones, dependiendo del archivo.

Por lo tanto, un EDF no contiene solamente una matriz de valores numéricos, sino también información descriptiva sobre la adquisición.

En este dataset, las grabaciones incluyen señales EEG, EOG, EMG y ECG.

Los canales observados son:

    EEG F4-M1
    EEG C4-M1
    EEG O2-M1
    EEG C3-M2
    EMG chin
    EOG E1-M2
    EOG E2-M2
    ECG

El pipeline principal utiliza cuatro canales EEG, dos EOG y un canal EMG.

---

## 1.3 Lectura de archivos EDF con MNE

El módulo `mne.io` contiene funciones para leer diferentes formatos de señales biomédicas.

Para archivos EDF se utiliza:

    raw = mne.io.read_raw_edf(
        sample_path,
        preload=False,
        verbose=False
    )

Esta función crea un objeto de tipo `Raw`.

El objeto `Raw` representa una grabación continua y reúne:

- las señales;
- los nombres de los canales;
- la frecuencia de muestreo;
- información temporal;
- metadatos de adquisición;
- métodos para acceder y procesar los datos.

---

## 1.4 Uso de `preload`

El parámetro:

    preload=False

indica que MNE no debe cargar inmediatamente todas las muestras del archivo en memoria.

Esto resulta útil cuando solamente se necesita:

- consultar metadatos;
- inspeccionar los canales;
- analizar una porción de la grabación;
- evitar cargar una noche completa de PSG innecesariamente.

Cuando se utiliza:

    preload=True

la señal completa se carga en memoria, lo que puede ser conveniente cuando se realizarán posteriormente operaciones sobre todo el registro, como filtrado o resampling.

---

## 1.5 El objeto `Raw`

`Raw` es un objeto específico de MNE diseñado para representar señales continuas.

Algunos de sus atributos más importantes son:

| Atributo | Descripción |
| --- | --- |
| `raw.ch_names` | Nombres de los canales |
| `raw.info` | Metadatos de la grabación |
| `raw.info["sfreq"]` | Frecuencia de muestreo en Hz |
| `raw.n_times` | Cantidad de muestras temporales |
| `raw.times` | Vector temporal en segundos |
| `raw.duration` | Duración de la grabación en segundos |

Por ejemplo:

    print(raw.ch_names)
    print(raw.info["sfreq"])
    print(raw.n_times)
    print(raw.duration)

---

## 1.6 Frecuencia de muestreo y vector temporal

Las grabaciones originales del dataset tienen una frecuencia de muestreo de:

    256 Hz

Esto significa que se registran 256 muestras por segundo y por canal.

El intervalo temporal entre dos muestras consecutivas es:

$$
\Delta t = \frac{1}{f_s}
$$

Para una frecuencia de muestreo de 256 Hz:

$$
\Delta t = \frac{1}{256} \approx 0.003906\ \text{s}
$$

Por lo tanto, los primeros valores de `raw.times` son aproximadamente:

    0.000000
    0.003906
    0.007812
    0.011719
    ...

La duración aproximada de la grabación también puede calcularse como:

$$
T \approx \frac{N}{f_s}
$$

donde:

- $N$ es la cantidad de muestras;
- $f_s$ es la frecuencia de muestreo.

En MNE puede obtenerse directamente mediante:

    raw.duration

---

## 1.7 Tipos de canal

MNE permite asociar un tipo a cada canal.

Los utilizados en este proyecto son:

| Tipo | Señal |
| --- | --- |
| `eeg` | Actividad eléctrica cerebral |
| `eog` | Movimientos oculares |
| `emg` | Actividad muscular |
| `ecg` | Actividad cardíaca |

Los tipos actualmente asignados pueden consultarse mediante:

    raw.get_channel_types()

Sin embargo, este método no analiza la forma de la señal para determinar automáticamente su origen fisiológico. Solamente devuelve el tipo que MNE tiene asignado en ese momento.

Como los nombres y tipos de los canales del dataset son conocidos, pueden asignarse explícitamente:

    channel_types = {
        "EEG F4-M1": "eeg",
        "EEG C4-M1": "eeg",
        "EEG O2-M1": "eeg",
        "EEG C3-M2": "eeg",
        "EMG chin": "emg",
        "EOG E1-M2": "eog",
        "EOG E2-M2": "eog",
        "ECG": "ecg",
    }

    raw.set_channel_types(channel_types)

Luego pueden verificarse con:

    for channel, channel_type in zip(
        raw.ch_names,
        raw.get_channel_types()
    ):
        print(f"{channel:<20} → {channel_type}")

---

## 1.8 Metadatos y features

Valores como:

    raw.ch_names
    raw.info["sfreq"]
    raw.n_times
    raw.times

describen la estructura de la grabación.

No deben confundirse con las **features** utilizadas posteriormente para entrenar modelos de machine learning.

Las features se obtienen a partir de las señales procesadas y pueden representar, por ejemplo:

- estadísticas temporales;
- potencia en bandas frecuenciales;
- características espectrales;
- medidas de complejidad;
- parámetros derivados de la morfología de la señal.

---

# 2. Análisis espectral de señales EEG

## 2.1 Dominio temporal y frecuencial

Una señal EEG puede analizarse desde diferentes representaciones.

En el dominio temporal se observa:

    amplitud vs. tiempo

Esto permite identificar:

- cambios de amplitud;
- eventos transitorios;
- artefactos;
- segmentos planos;
- morfología de la señal.

En el dominio frecuencial se estudia:

    potencia vs. frecuencia

Esto permite analizar qué componentes frecuenciales predominan en la señal.

Ambas representaciones son complementarias.

---

## 2.2 Power Spectral Density

La **Power Spectral Density** o **PSD** representa cómo se distribuye la potencia de una señal entre distintas frecuencias.

Conceptualmente:

    señal temporal
        ↓
    transformación al dominio frecuencial
        ↓
    potencia asociada a cada frecuencia
        ↓
    PSD

La PSD permite responder preguntas como:

- ¿en qué frecuencias se concentra la actividad de la señal?;
- ¿predominan componentes lentas o rápidas?;
- ¿existen diferencias espectrales entre canales?;
- ¿se observan patrones asociados con diferentes etapas del sueño?

---

# 3. Transformada de Fourier

La **Transformada de Fourier** permite representar una señal temporal como una combinación de componentes frecuenciales.

Para una señal discreta:

$$
x[n]
$$

la transformada genera valores:

$$
X[k]
$$

donde cada índice $k$ representa una frecuencia determinada.

La frecuencia asociada al índice $k$ depende de la frecuencia de muestreo y del número de puntos utilizado en la FFT:

$$
f_k = \frac{k f_s}{N_{FFT}}
$$

La separación entre bins frecuenciales es:

$$
\Delta f = \frac{f_s}{N_{FFT}}
$$

Por ejemplo, utilizando:

    fs = 256 Hz
    n_fft = 1024

se obtiene:

$$
\Delta f = \frac{256}{1024} = 0.25\ \text{Hz}
$$

Esto significa que el espectro contiene valores separados aproximadamente cada 0.25 Hz.

---

# 4. PSD mediante el método de Welch

En este proyecto se utiliza principalmente el método de **Welch** para estimar la PSD.

Conceptualmente, Welch realiza:

    señal
        ↓
    dividir en segmentos temporales
        ↓
    aplicar una ventana a cada segmento
        ↓
    calcular FFT
        ↓
    obtener potencia por frecuencia
        ↓
    repetir para todos los segmentos
        ↓
    promediar
        ↓
    PSD

Para cada segmento temporal $m$ se obtiene un espectro:

$$
X_m[k]
$$

y su potencia es proporcional a:

$$
|X_m[k]|^2
$$

Posteriormente, Welch promedia las estimaciones obtenidas en los distintos segmentos.

De forma conceptual:

$$
PSD(f_k) =
\frac{1}{M}
\sum_{m=1}^{M}
P_m(f_k)
$$

donde:

- $M$ es la cantidad de segmentos;
- $P_m(f_k)$ es la estimación de potencia de la frecuencia $f_k$ en el segmento $m$.

El promedio reduce la variabilidad de una única FFT y produce una estimación espectral más estable.

---

## 4.1 Cálculo con MNE

Una PSD puede calcularse mediante:

    spectrum = raw_psd.compute_psd(
        method="welch",
        fmin=0.5,
        fmax=35,
        n_fft=1024,
        picks=eeg_channels
    )

Los principales parámetros son:

### `method="welch"`

Indica que se utilizará el método de Welch para estimar la densidad espectral de potencia.

### `fmin=0.5`

Limita el análisis inferiormente a 0.5 Hz.

### `fmax=35`

Limita el análisis superiormente a 35 Hz.

Este es el rango espectral utilizado en el baseline del proyecto.

### `n_fft=1024`

Determina la cantidad de puntos utilizados para calcular la FFT.

Con una frecuencia original de 256 Hz:

$$
\Delta f = \frac{256}{1024} = 0.25\ \text{Hz}
$$

### `picks=eeg_channels`

Permite calcular la PSD únicamente para los canales EEG seleccionados.

En este proyecto:

    eeg_channels = [
        "EEG F4-M1",
        "EEG C4-M1",
        "EEG O2-M1",
        "EEG C3-M2"
    ]

---

# 5. Bandas frecuenciales de EEG

Para interpretar el espectro se consideran diferentes bandas de frecuencia.

Los rangos utilizados en este proyecto son aproximadamente:

| Banda | Rango |
| --- | --- |
| Delta | 0.5–4 Hz |
| Theta | 4–8 Hz |
| Alpha | 8–13 Hz |
| Sigma | 13–15 Hz |
| Beta | 15–30 Hz |

Estas bandas permiten resumir parte de la información espectral de la señal EEG.

---

## 5.1 Delta

La banda delta comprende principalmente frecuencias bajas:

    0.5–4 Hz

La actividad delta aumenta especialmente durante el sueño profundo y es una característica importante de la etapa N3.

---

## 5.2 Theta

La banda theta se encuentra aproximadamente entre:

    4–8 Hz

Puede observarse durante etapas de sueño ligero, especialmente N1, y también puede estar presente durante REM.

---

## 5.3 Alpha

La banda alpha comprende aproximadamente:

    8–13 Hz

Es característica de estados de vigilia relajada, especialmente con los ojos cerrados.

---

## 5.4 Sigma

La banda sigma utilizada en este proyecto se concentra aproximadamente entre:

    13–15 Hz

Está relacionada con la presencia de **sleep spindles**, eventos transitorios característicos de la etapa N2.

Los spindles no representan una etapa del sueño en sí mismos, sino un patrón EEG utilizado como indicador de N2.

---

## 5.5 Beta

La banda beta se encuentra aproximadamente entre:

    15–30 Hz

Está asociada con actividad cerebral relativamente rápida y suele observarse con mayor presencia durante vigilia.

---

# 6. Interpretación de la PSD en sleep staging

Las etapas del sueño no se identifican utilizando exclusivamente una banda frecuencial.

Por ejemplo:

- N3 suele presentar mayor actividad lenta;
- N2 puede presentar actividad sigma asociada con spindles;
- Wake puede mostrar actividad alpha y mayor contenido de frecuencias rápidas;
- N1 y REM pueden presentar espectros EEG relativamente similares.

REM, en particular, no posee una banda EEG exclusiva que permita identificarlo de forma aislada.

Su clasificación se beneficia especialmente de combinar:

- EEG;
- movimientos oculares registrados mediante EOG;
- reducción del tono muscular registrada mediante EMG.

Por esta razón, el proyecto utiliza información multicanal y no solamente características espectrales del EEG.

---

# 7. PSD promedio por sleep stage

También es posible calcular una PSD individual para cada epoch de 30 segundos y posteriormente agrupar los resultados según la etapa del sueño.

Para una etapa $s$:

$$
\overline{PSD}_s(f)
=
\frac{1}{N_s}
\sum_{i=1}^{N_s}
PSD_i(f)
$$

donde $N_s$ representa la cantidad de epochs pertenecientes a esa etapa.

Esto permite comparar las tendencias espectrales promedio de:

    Wake
    N1
    N2
    N3
    REM

Las curvas no necesariamente presentan separaciones muy marcadas.

Esto puede ocurrir porque:

- todas las señales EEG comparten una estructura espectral de fondo;
- el promedio suaviza eventos transitorios;
- algunos patrones característicos aparecen solamente durante períodos breves;
- diferentes etapas pueden compartir componentes frecuenciales;
- la clasificación clínica utiliza información temporal, morfológica y multicanal.

Por este motivo, la PSD debe interpretarse como una herramienta descriptiva y no como un criterio único de clasificación.

---

# 8. Spectrogramas

Mientras que una PSD convencional representa:

$$
PSD(f)
$$

un spectrograma representa cómo cambia el contenido frecuencial a lo largo del tiempo:

$$
S(f,t)
$$

Por lo tanto, incorpora dos dimensiones:

    frecuencia
    +
    tiempo

Conceptualmente:

    señal continua
        ↓
    ventanas temporales sucesivas
        ↓
    espectro de cada ventana
        ↓
    matriz frecuencia × tiempo
        ↓
    spectrograma

Esto permite observar cambios en la actividad espectral durante la noche y relacionarlos con la evolución del hypnogram.

---

# 9. Relación con el pipeline del proyecto

El flujo general utilizado en el proyecto puede resumirse como:

    archivos EDF
        ↓
    lectura con MNE
        ↓
    señales EEG + EOG + EMG
        ↓
    análisis exploratorio
        ├── dominio temporal
        ├── calidad de señal
        ├── PSD
        └── spectrogramas
        ↓
    preprocessing
        ├── selección de canales
        ├── control de calidad
        ├── resampling
        ├── filtrado
        ├── alineación con labels
        ├── segmentación en epochs
        └── normalización
        ↓
    datos procesados
        ↓
    dos posibles líneas de modelado
        ├── features + modelos clásicos / boosting
        └── señal procesada + deep learning

El análisis exploratorio permite comprender la estructura de las señales y detectar problemas de calidad antes de aplicar el procesamiento de forma automática sobre todo el dataset.

El procesamiento posterior transforma cada epoch de 30 segundos en una representación multicanal uniforme, manteniendo EEG, EOG y EMG para las etapas posteriores de modelado.
```
