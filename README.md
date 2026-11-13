# Automatic Sleep Staging

Trabajo final de la Carrera de Especialización en Inteligencia Artificial (FIUBA).

Clasificación automática de etapas del sueño (W, N1, N2, N3 y REM) a partir de señales polisomnográficas multicanal EEG, EOG y EMG. Se comparan modelos tabulares, una CNN unidimensional por intervalo y un modelo temporal CNN + BiGRU.

## Datos y protocolo

- Base original: HMC Sleep Staging Database, versión 1.1 (PhysioNet), con 151 registros.
- Después del procesamiento y control de calidad: **149 sujetos**, **135.471 intervalos** de 30 segundos, **7 canales** y **186 características tabulares** por intervalo.
- Separación por sujeto: **119 sujetos** de desarrollo (95 entrenamiento y 24 validación para redes) y **30 sujetos** de prueba.
- Comparación temporal: se usan **26.446 centros de prueba** comunes, válidos para una ventana de 7 intervalos; los límites y discontinuidades no se rellenan artificialmente.
- Métrica principal: **F1 macro**. Las configuraciones se eligen con **validación**, no con test.

Los datos clínicos originales, archivos intermedios grandes y pesos entrenados no se incluyen necesariamente en el repositorio. Para ejecutar experimentos posteriores se necesitan los artefactos producidos en las etapas previas.

## Notebooks disponibles en GitHub

Dentro de `Notebooks/`:

| Notebook | Contenido |
| --- | --- |
| `01_data_exploration.ipynb` | Exploración de registros y distribución de etapas |
| `02_preprocessing.ipynb` | Selección de canales, alineación, control de calidad, filtrado y remuestreo |
| `03_feature_engineering.ipynb` | Extracción de características temporales y espectrales |
| `04_classical_baseline.ipynb` | Modelos clásicos de referencia |
| `05_gradient_boosting_models.ipynb` | Comparación de XGBoost, LightGBM y CatBoost |
| `06_deep_learning_cnn_baseline.ipynb` | 1D-CNN sobre intervalos multicanal |
| `07_model_comparison_and_error_analysis.ipynb` | Comparación inicial y análisis de errores |
| `08_temporal_context.ipynb` | Modelo temporal CNN + BiGRU |
| `09_context_size_sensitivity.ipynb` | Ventanas temporales de 3, 5 y 7 intervalos |
| `10_end_to_end_finetuning.ipynb` | Evaluación del ajuste conjunto CNN + BiGRU |
| `11_final_comparison_and_model_selection.ipynb` | Comparación y **selección preliminar** basada en una corrida |

> **Importante sobre NB11:** la notebook publicada documenta un experimento inicial que seleccionó una ventana de **5 intervalos** (F1 macro de validación de 0,7816). No debe interpretarse como el resultado final actualizado. La memoria V4.4 incluye experimentos adicionales con cinco semillas que llevan a una selección de **7 intervalos**. Los resultados históricos de NB11 deben conservarse; la actualización de selección corresponde a NB12. Si se vuelve a ejecutar la NB11 antigua, puede reescribir `reports/results/final_model_selection.json` con el resultado preliminar.

## Experimentos posteriores documentados en la memoria V4.4

Estos archivos **todavía están pendientes de publicación en GitHub**:

- `12_multiseed_robustness.ipynb`: cinco semillas (42, 123, 2026, 7, 99), comparación de ventanas y evaluación de robustez del modelo temporal.
- `13_error_analysis_and_api_validation.ipynb`: análisis de confusiones y transiciones, estudios por sujeto y comprobaciones de la API local FastAPI con entradas NPZ ya procesadas.

La memoria V4.4 contiene los resultados de estas etapas. Su publicación en el repositorio permitirá reproducir y consultar el análisis experimental y la validación de integración.

### Resultados de referencia de la memoria V4.4

| Modelo | F1 macro en test (26.446 centros comunes) |
| --- | ---: |
| Random Forest | 0,7000 |
| LightGBM | 0,7210 |
| CNN por intervalo | 0,7348 |
| CNN congelada + BiGRU, 7 intervalos (media de 5 semillas) | **0,7749 ± 0,0018** |

La longitud de **7 intervalos (210 segundos)** presentó el mayor F1 macro medio de **validación** entre las longitudes probadas (0,7814, frente a 0,7788 con 5 intervalos y 0,7741 con 3). La diferencia entre 7 y 5 fue pequeña en relación con la variación entre semillas; estos ensayos no demuestran superioridad estadística entre ambos contextos. El fine-tuning conjunto solo se evaluó con contexto de 5 intervalos y no mejoró la media de validación.

Se observó una mejora media por sujeto en 29 de los 30 registros de prueba, aunque SN069 empeoró en las cinco semillas. N1 continuó siendo la etapa más difícil. Estas conclusiones se limitan a un encoder CNN y partición por sujeto fijos, sin evaluación externa.

## Integración del modelo

Se desarrolló un prototipo local con FastAPI que recibe archivos `.npz` ya procesados, normaliza con estadísticas de entrenamiento y aplica CNN congelada + BiGRU. Expone:

- `GET /`: interfaz web de carga, hipnograma y exportación de CSV/JSON.
- `GET /health`: disponibilidad del servicio.
- `POST /predict`: inferencia sobre centros válidos y trazabilidad.
- `GET /docs`: documentación interactiva.

En la prueba documentada sobre SN069, el prototipo reprodujo **731 de 731** etiquetas de la notebook con semilla 42 en tres solicitudes consecutivas. No realiza aún todo el procesamiento desde EDF crudo ni una validación clínica.

## Organización

```text
Automatic-Sleep-Staging/
├── Notebooks/
├── data/                   # parte de los datos grandes no se versiona
├── models/                 # checkpoints locales
├── reports/
│   ├── figures/
│   └── results/
├── docs/
└── README.md
```

**Estado de sincronización (13 de noviembre de 2026):** README alineado con la memoria V4.4, pero resta publicar y comprobar las versiones finales de NB12 y NB13, y actualizar la NB11 para evitar que sobrescriba la selección final posterior.
