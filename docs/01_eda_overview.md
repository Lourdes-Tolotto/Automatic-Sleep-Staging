# Notebook 01 — Exploratory Data Analysis

Este documento resume el objetivo, las etapas principales y las decisiones derivadas de la notebook `01_eda_sleep_staging.ipynb`.

La notebook realiza el análisis exploratorio inicial del dataset de polisomnografía antes de aplicar el pipeline definitivo de preprocessing.

El objetivo principal no es transformar todavía los datos para el entrenamiento, sino comprender su estructura, verificar su consistencia, analizar las señales y detectar posibles problemas de calidad que deban considerarse en las etapas posteriores.

Para una explicación más detallada de conceptos como MNE, archivos EDF, PSD, FFT y método de Welch, consultar [`eda_background.md`](eda_background.md).

---

# 1. Objetivos de la notebook

La EDA busca responder principalmente las siguientes preguntas:

- ¿Cuántos recordings y sujetos contiene el dataset?
- ¿Todos los archivos presentan la misma estructura?
- ¿Qué canales biomédicos están disponibles?
- ¿Cuál es la frecuencia de muestreo?
- ¿Cuánto duran las grabaciones?
- ¿Los archivos de señales y labels están correctamente asociados?
- ¿Cómo se distribuyen las etapas del sueño?
- ¿Qué características temporales y espectrales presentan las señales?
- ¿Existen artefactos o recordings con problemas importantes de calidad?
- ¿Hay sujetos que deberían excluirse antes del preprocessing?

Las decisiones obtenidas en esta notebook se utilizan posteriormente en `02_preprocessing.ipynb`.

---

# 2. Archivos utilizados

El dataset contiene principalmente dos tipos de archivos por sujeto:

- un archivo `.edf` con las señales biomédicas continuas;
- un archivo `_sleepscoring.txt` con las anotaciones de sleep stages.

Los recordings se encuentran en:

    data/recordings/

Cada sujeto se identifica mediante un código como:

    SN001
    SN002
    SN003
    ...

La notebook verifica que cada archivo de señal tenga disponible su correspondiente archivo de labels.

---

# 3. Estructura de las grabaciones

Los archivos EDF contienen ocho canales:

| Canal | Modalidad |
| --- | --- |
| EEG F4-M1 | EEG |
| EEG C4-M1 | EEG |
| EEG O2-M1 | EEG |
| EEG C3-M2 | EEG |
| EOG E1-M2 | EOG |
| EOG E2-M2 | EOG |
| EMG chin | EMG |
| ECG | ECG |

Durante la EDA se inspeccionan las señales disponibles, aunque el pipeline principal utilizará posteriormente siete canales:

- 4 EEG;
- 2 EOG;
- 1 EMG.

El ECG se mantiene fuera del baseline inicial.

---

# 4. Verificación de consistencia del dataset

Antes de analizar las señales se verifica la estructura global de todos los recordings.

Para cada archivo se revisan:

- cantidad de canales;
- nombres de canales;
- frecuencia de muestreo;
- duración de la grabación;
- cantidad de epochs disponibles;
- cantidad de labels.

El dataset analizado contiene `151` recordings.

Todos los archivos presentan:

    8 canales
    frecuencia de muestreo = 256 Hz

La duración de las grabaciones varía entre sujetos, lo cual es esperable en registros de sueño nocturno.

También se verifica que la composición de canales sea consistente entre los recordings.

Esta comprobación es importante porque las etapas posteriores del pipeline requieren una representación multicanal uniforme.

---

# 5. Inspección de una grabación de ejemplo

Se utiliza un sujeto de ejemplo para explorar la estructura de los archivos EDF y visualizar las señales.

La inspección incluye:

- nombres de los canales;
- tipos de canal;
- frecuencia de muestreo;
- duración;
- cantidad total de muestras;
- segmentos temporales de EEG, EOG, EMG y ECG.

Se generan visualizaciones tanto de señales normalizadas como de amplitudes expresadas en microvoltios.

Las señales normalizadas permiten comparar más fácilmente la forma temporal de canales con escalas diferentes.

Las señales sin normalizar permiten evaluar amplitudes reales y detectar posibles valores extremos, discontinuidades o saturaciones.

---

# 6. Análisis de los sleep stages

Los archivos de scoring contienen las etapas del sueño asociadas a epochs de 30 segundos.

Las clases consideradas son:

| Stage | Descripción |
| --- | --- |
| W | Wake |
| N1 | Sueño NREM ligero |
| N2 | Sueño NREM |
| N3 | Sueño profundo |
| R | REM |

Se construye un hypnogram para visualizar la evolución de las etapas durante una noche.

El hypnogram permite observar las transiciones entre Wake, NREM y REM y comprobar que las anotaciones presentan una estructura temporal coherente.

---

# 7. Distribución global de clases

Se combinan los labels de todos los sujetos para estudiar la distribución de sleep stages en el dataset.

Para cada clase se calculan:

- cantidad de epochs;
- porcentaje respecto del total;
- relación entre la clase más frecuente y la menos frecuente.

El análisis muestra que las clases no están distribuidas uniformemente.

N2 presenta una presencia importante en el dataset, mientras que otras etapas, especialmente N1, cuentan con una cantidad menor de ejemplos.

Este desbalance debe tenerse en cuenta posteriormente durante:

- la selección de métricas;
- la validación;
- el entrenamiento de los modelos;
- la posible utilización de class weights.

En consecuencia, la accuracy no debe utilizarse como única métrica de evaluación.

---

# 8. Análisis temporal de las señales

Se inspeccionan segmentos temporales de los recordings para observar:

- amplitudes;
- oscilaciones;
- diferencias entre modalidades;
- artefactos;
- regiones planas;
- eventos transitorios.

EEG, EOG y EMG contienen información complementaria para sleep staging.

El EEG permite analizar principalmente la actividad cerebral y sus patrones temporales y frecuenciales.

El EOG aporta información sobre movimientos oculares y resulta especialmente relevante para REM.

El EMG permite observar cambios en el tono muscular, también importantes para diferenciar REM de otras etapas.

---

# 9. Análisis espectral

La notebook incluye un análisis espectral exploratorio de los canales EEG.

Se calcula la Power Spectral Density mediante el método de Welch en el rango:

    0.5–35 Hz

Se consideran las siguientes bandas aproximadas:

| Banda | Rango |
| --- | --- |
| Delta | 0.5–4 Hz |
| Theta | 4–8 Hz |
| Alpha | 8–13 Hz |
| Sigma | 13–15 Hz |
| Beta | 15–30 Hz |

La PSD permite observar cómo se distribuye la potencia de las señales EEG entre distintas frecuencias.

También se calcula una PSD promedio por sleep stage para analizar posibles diferencias espectrales entre Wake, N1, N2, N3 y REM.

El análisis muestra que existen tendencias asociadas con distintas etapas, pero las curvas presentan superposición.

Esto es esperable porque el sleep staging no depende exclusivamente del contenido frecuencial del EEG, sino también de características temporales, morfológicas y de otras modalidades como EOG y EMG.

---

# 10. Spectrograma

Además de la PSD promedio, se utiliza un spectrograma para estudiar cómo cambia el contenido frecuencial del EEG a lo largo del tiempo.

Mientras que la PSD representa principalmente:

    potencia vs. frecuencia

el spectrograma representa:

    potencia vs. frecuencia y tiempo

Esto permite relacionar cambios espectrales con la evolución temporal del hypnogram.

El análisis conjunto de hypnogram y spectrograma permite observar que la actividad frecuencial del EEG cambia a lo largo de la noche y acompaña las distintas etapas del sueño.

---

# 11. Control exploratorio de calidad de señal

La notebook realiza un primer screening de calidad sobre los canales EEG.

Este análisis no constituye todavía el criterio definitivo de descarte utilizado en preprocessing.

Su objetivo es detectar recordings o epochs que merezcan una inspección adicional.

Se consideran inicialmente dos indicadores:

### Amplitud elevada

Un epoch se marca como sospechoso cuando al menos uno de sus canales EEG presenta:

    |amplitud| > 500 µV

Este valor se utiliza como umbral exploratorio y no como criterio clínico universal de descarte.

### Segmentos planos

También se identifican períodos donde la señal mantiene valores idénticos durante al menos:

    1 segundo

Estos casos pueden indicar pérdida de contacto, congelamiento de la señal u otros problemas de adquisición.

---

# 12. Calidad de señal por sujeto

Los resultados de calidad se resumen a nivel de sujeto.

Para cada recording se calcula:

- cantidad total de epochs;
- cantidad de epochs sospechosos;
- porcentaje de epochs sospechosos.

Esto permite identificar recordings cuyo comportamiento se diferencia claramente del resto del dataset.

La mayoría de los sujetos conserva una proporción elevada de señal utilizable.

Sin embargo, algunos recordings presentan porcentajes considerablemente mayores de epochs marcados.

---

# 13. Revisión de recordings extremos

Como criterio exploratorio, los sujetos con más del `40%` de epochs sospechosos se seleccionan para una inspección adicional.

Este porcentaje se utiliza únicamente para identificar casos extremos que requieren revisión y no constituye un umbral automático de exclusión.

El análisis identificó principalmente:

    SN032 → 95.5% de epochs sospechosos
    SN033 → 81.0% de epochs sospechosos

También se estudió `SN020`, que presenta un porcentaje elevado pero menos extremo:

    SN020 → 42.2% de epochs sospechosos

Para estos sujetos se analizó:

- qué canales producen los flags;
- cantidad de epochs con amplitudes elevadas;
- presencia de segmentos planos;
- distribución temporal de los problemas;
- amplitudes máximas observadas.

---

# 14. Identificación de clipping

La inspección detallada mostró que una gran cantidad de muestras de algunos recordings alcanzan exactamente:

    +800 µV
    -800 µV

La repetición de estos valores exactos es compatible con un fenómeno de clipping o saturación de la señal.

En `SN032`, los cuatro canales EEG presentan clipping de forma extensa y los problemas aparecen durante prácticamente toda la grabación.

En `SN033` también se observa clipping multicanal distribuido a lo largo de gran parte del recording.

Estos patrones indican problemas sistemáticos de calidad y no solamente artefactos aislados.

---

# 15. Caso SN020

`SN020` presenta un comportamiento diferente.

El sujeto tiene:

    948 epochs totales
    400 epochs sospechosos
    42.2% de epochs sospechosos

La mayor parte de los eventos de amplitud elevada afecta simultáneamente a:

    EEG F4-M1
    EEG C4-M1
    EEG O2-M1

Estos tres canales comparten la referencia `M1`.

En cambio:

    EEG C3-M2

utiliza la referencia `M2` y no presenta clipping en ±800 µV.

Este patrón es compatible con un posible problema asociado con la referencia M1, aunque esta interpretación se considera una hipótesis y no una conclusión definitiva sobre el sistema de adquisición.

A pesar de los epochs afectados, SN020 conserva aproximadamente `4.6 horas` de señal no marcada.

Por este motivo, el recording se mantiene en el dataset y los epochs problemáticos se gestionarán individualmente durante el preprocessing.

---

# 16. Decisión sobre recordings problemáticos

A partir del análisis exploratorio y de la inspección específica de los casos extremos, se toma la siguiente decisión:

| Sujeto | Decisión |
| --- | --- |
| SN032 | Excluir |
| SN033 | Excluir |
| SN020 | Conservar |

SN032 y SN033 se excluyen porque presentan alteraciones extensas, multicanal y sostenidas durante gran parte de las grabaciones.

La exclusión no se basa únicamente en superar un porcentaje arbitrario de epochs sospechosos.

El porcentaje elevado se utiliza primero para identificar los casos y posteriormente la decisión se confirma mediante el análisis de amplitudes, canales afectados, distribución temporal y evidencia de clipping.

SN020 se conserva porque mantiene una cantidad considerable de señal utilizable y el patrón de alteración es más localizado.

---

# 17. Decisiones transferidas al preprocessing

Las principales decisiones de la EDA que afectan a `02_preprocessing.ipynb` son:

1. utilizar cuatro canales EEG, dos EOG y un EMG;
2. excluir el ECG del baseline inicial;
3. mantener una representación multicanal uniforme;
4. trabajar posteriormente con epochs de 30 segundos;
5. utilizar el rango espectral de interés de 0.5–35 Hz;
6. reducir posteriormente la frecuencia de muestreo de 256 Hz a 100 Hz;
7. realizar control de calidad antes de generar el dataset definitivo;
8. excluir los recordings SN032 y SN033;
9. conservar SN020 y gestionar sus problemas a nivel de epoch.

Estas decisiones permiten que la etapa de preprocessing se base en observaciones realizadas previamente sobre los datos y no únicamente en parámetros definidos de forma arbitraria.

---

# 18. Rol de la NB1 dentro del proyecto

La notebook 01 puede resumirse mediante el siguiente flujo:

    archivos EDF + sleep scoring
                ↓
    inspección de estructura
                ↓
    consistencia entre sujetos
                ↓
    análisis de labels
                ↓
    distribución de sleep stages
                ↓
    visualización temporal
                ↓
    análisis espectral
                ↓
    control exploratorio de calidad
                ↓
    inspección de recordings problemáticos
                ↓
    decisiones para preprocessing

La salida principal de esta notebook no es un nuevo dataset procesado.

Su resultado es un conjunto de observaciones y decisiones metodológicas que sirven como entrada conceptual para `02_preprocessing.ipynb`.

A partir de la NB2 comienza la transformación sistemática de los recordings crudos en epochs multicanal preparados para feature engineering y modelado.