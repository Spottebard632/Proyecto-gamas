# Clasificación de CPU, GPU y coolers en gamas de rendimiento

Proyecto final de **Aprendizaje de Máquina** · Autores: **David Cediel** y **David Gómez**


## Descripción

El proyecto clasifica **procesadores (CPU), tarjetas gráficas (GPU) y coolers** en **gamas de rendimiento** usando únicamente las características de catálogo del producto (memoria, relojes, núcleos, TDP, peso, tipo de refrigeración, etc.) y **sin usar el benchmark** con el que se construyó la etiqueta.

- En **CPU y GPU** la gama se construye con puntuaciones de benchmark (*CPU Mark* y *G3D Mark*).
- En **coolers** la gama se construye con las mediciones de **Cybenetics** (calificación THETA).

El resultado no reemplaza al benchmark: mide qué tan bien se puede **aproximar la gama** a partir de las especificaciones.

## ¿Qué se hizo en cada avance?

| Avance | Contenido | Secciones del notebook |
|---|---|---|
| **1** | Construcción de los datasets, definición de las gamas y análisis exploratorio (EDA) | 1 a 3 |
| **2** | Clasificación en **cinco gamas** con GNB, árbol de decisión, Random Forest y SVM: Fase 1 (hiperparámetros), Fase 2 (validación cruzada), Fase 3 (gráficas) y test final | 4 a 8 |
| **3** | **A.** Clasificación **binaria** (gama baja vs. gama alta) · **B.** **Regresión** del puntaje del benchmark · elección de los **2 mejores modelos** de cada tipo | 9 a 11 |

## Estructura del repositorio

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── README.md                      # descripción de los CSV
│   ├── coolers_amd_am5.csv            # Cybenetics, plataforma AMD AM5
│   └── coolers_intel_lga1851.csv      # Cybenetics, plataforma Intel LGA1851
├── notebooks/
│   └── Avance_3_ML_y_Terminado.ipynb  # notebook completo con sus salidas
├── src/
│   └── proyecto_gamas.py              # solo el código del notebook, en el mismo orden
└── docs/
    └── informe.md                     # solo el texto explicativo del notebook
```

## Datos

Los catálogos de GPU/CPU/coolers y los benchmarks de CPU/GPU **se descargan automáticamente desde GitHub** al ejecutar el código. Los **dos CSV de coolers** (Cybenetics) están en la carpeta [`data/`](data/) y se leen desde ahí.

| Dataset | Origen | Filas finales | Variable a predecir |
|---|---|---:|---|
| **GPU** | Catálogo `video-card.csv` + puntaje **G3D Mark** | 5.513 | `gama` (5 clases) · `benchmark_score` (regresión) |
| **CPU** | Catálogo `cpu.csv` + puntaje **CPU Mark** | 627 | `gama` (5 clases) · `benchmark_score` (regresión) |
| **COOLER** | CSV de Cybenetics (AM5 + LGA1851) + catálogo `cpu-cooler.csv` | 48 | `gama` (5 clases) · `score_percentile` (regresión) |

**Cómo se construyen las gamas.** Los modelos se ordenan por su puntuación y se dividen en **cinco grupos del mismo tamaño** (baja, media-baja, media, media-alta y alta). En los coolers, como AMD e Intel se miden por separado, primero se calcula el percentil de *Overall performance* dentro de cada plataforma y después se promedia. En GPU se conserva cada variante comercial, por eso las gamas quedan desbalanceadas por filas (751 / 846 / 989 / 1.203 / 1.724).

**Variables de entrada.**
- **GPU:** precio, memoria, reloj base, reloj boost, color y longitud.
- **CPU:** precio, núcleos, reloj base, reloj boost, TDP, microarquitectura y gráficos integrados.
- **COOLER:** tipo de refrigeración, dimensiones, peso y número de ventiladores (Cybenetics), más precio, rpm, nivel de ruido, tamaño y color del catálogo cuando el modelo se logra relacionar.

Se **excluyen** la gama, los nombres/identificadores y las puntuaciones con las que se creó la etiqueta (`benchmark_score`, `score_percentile`) para evitar fuga de información.

## Metodología

- **Partición 60 % train / 20 % validación / 20 % test**, estratificada por gama y **agrupada por `match_key`** (un mismo modelo o chipset nunca queda en dos conjuntos). Semilla fija: `SEED = 21`.
- **Fuga de información:** se compara el F1 con las variables permitidas contra el F1 al añadir variables que ayudaron a crear la etiqueta (diagnóstico, no se usa para elegir modelos).
- **Preprocesamiento dentro del `Pipeline`:** imputación con la mediana (aprendida solo en train), `StandardScaler` en SVM/SVR y `pd.get_dummies` para variables categóricas.
- **Clasificación:** GNB, árbol de decisión (DT), Random Forest (RF) y SVM, más un modelo base que siempre responde la clase más frecuente. **170 configuraciones por dataset.** Métricas con promedio **macro** (Accuracy, Precision, Recall, F1).
- **Selección en dos fases:** Fase 1 en validación (mejor configuración de cada algoritmo) y Fase 2 con **validación cruzada K-Fold** sobre train (K = 5 o menos si una gama tiene pocos grupos, **10 repeticiones** con semillas distintas). El modelo final se **reentrena con train + validación** y se evalúa **una sola vez** en test.
- **Regresión:** modelo lineal / Ridge, árbol, Random Forest y SVR dentro de un `TransformedTargetRegressor`. Métricas: MAE, RMSE y R².

## Resultados

Las cifras de esta sección son las de las **salidas guardadas en el notebook** (conjunto de test).

### Avance 2 — cinco gamas

| Dataset | Modelo elegido por validación cruzada | Accuracy | F1 macro | Test (n) |
|---|---|---:|---:|---:|
| GPU | RF (`n_estimators=200`, `max_depth=8`, `class_weight='balanced'`) | 0,729 | 0,683 | 1.085 |
| CPU | RF (`n_estimators=100`, sin límite de profundidad) | 0,825 | 0,821 | 126 |
| COOLER | SVM RBF (`C=1`, `gamma=0.01`, `class_weight='balanced'`) | 0,300 | 0,167 | 10 |

### Avance 3 · Parte A — clasificación binaria (gama baja vs. gama alta)

La gama media se **descarta** y las gamas de los extremos se agrupan. Los puestos se asignan por validación cruzada.

| Dataset | Puesto 1 | F1 test | Puesto 2 | F1 test | Modelo base (F1) |
|---|---|---:|---|---:|---:|
| GPU | SVM lineal (`C=0.1`, balanced) | 0,998 | DT (`max_depth=1`) | 1,000 | 0,392 |
| CPU | SVM lineal (`C=1`) | 1,000 | DT (`max_depth=8`) | 0,980 | 0,331 |
| COOLER | DT (`max_depth=1`) | 1,000 | RF (`n_estimators=50`, `max_depth=4`) | 1,000 | 0,333 |

> **Ojo:** estos valores son casi perfectos porque al quitar la gama media solo quedan los extremos, que se separan con reglas muy simples. Un árbol de **un solo corte** logra 100 % en GPU (`memory` ≤ 5 GB), 87 % en CPU (`core_count` ≤ 3,5) y 100 % en cooler (líquido vs. aire). No es fuga de información, pero el resultado es **optimista**.

### Avance 3 · Parte B — regresión del puntaje

| Dataset | Puesto 1 | R² test | MAE test | Puesto 2 | R² test | MAE test |
|---|---|---:|---:|---|---:|---:|
| GPU (G3D Mark) | RF (`n=200`, `max_features='sqrt'`) | 0,914 | 2.482 | Ridge (`alpha=100`) | 0,905 | 2.663 |
| CPU (CPU Mark) | Ridge (`alpha=1`) | 0,956 | 982 | RF (`n=200`, `max_features='sqrt'`) | 0,968 | 768 |
| COOLER (percentil) | DT (`max_depth=2`, `min_samples_leaf=5`) | 0,629 | 0,136 | RF (`n=100`, `max_depth=8`) | 0,687 | 0,117 |

El modelo base (predecir el promedio) obtiene R² ≈ 0 en los tres datasets. En CPU el segundo puesto tuvo mejor R² en test, pero la diferencia en validación cruzada era menor que la desviación entre folds, así que **no se cambió la elección tras ver el test**.

### Otros hallazgos

- **Fuga de información (F1 macro en validación, RF):** al añadir las variables que ayudaron a crear la etiqueta, el F1 sube de 0,652 a 0,996 en GPU, de 0,801 a 0,944 en CPU y de 0,267 a 0,593 en cooler. Por eso esas variables se excluyen de los predictores.
- **Variables más importantes (Random Forest de regresión):** GPU → `memory`, `core_clock`, `boost_clock`; CPU → `core_count`, `tdp`, `boost_clock`; cooler → `peso_g`, tipo de refrigeración y número de ventiladores.
- **Gama media (no vista):** el mejor modelo binario manda el 68 % de las 105 GPU de gama media a *gama baja* y el 68 % de las 25 CPU de gama media a *gama alta*.

## Cómo ejecutarlo

### Opción A — Google Colab (recomendada)

1. Abre el notebook con el botón **Abrir en Colab** de arriba.
2. *Entorno de ejecución → Ejecutar todo*.
3. Cuando lo pida, sube los dos archivos de [`data/`](data/): `coolers_amd_am5.csv` y `coolers_intel_lga1851.csv`.

### Opción B — En tu computador (script)

```bash
git clone https://github.com/TU_USUARIO/clasificacion-gamas-cpu-gpu-coolers.git
cd clasificacion-gamas-cpu-gpu-coolers

python -m venv .venv
source .venv/bin/activate          # en Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/proyecto_gamas.py
```

- Necesita **conexión a internet** (descarga los catálogos y benchmarks).
- La ejecución completa entrena 170 configuraciones de clasificación por dataset (más validación cruzada y regresión), así que puede tardar varios minutos o más según el computador.
- Las tablas se imprimen en la consola y las **gráficas se guardan** en `outputs/figuras/`.
- Si prefieres abrir el notebook localmente con Jupyter, copia los dos CSV a la misma carpeta del notebook.

> Los resultados usan semillas fijas, pero pueden cambiar en los decimales con otra versión de scikit-learn.
> El script se probó con Python 3.12, pandas 2.3, numpy 2.5, scikit-learn 1.9 y seaborn 0.13.

## Limitaciones

- **Coolers tiene muy pocos datos:** 48 modelos (28 / 10 / 10 en train / validación / test). Un solo producto mueve mucho las métricas, así que esos resultados son **preliminares**.
- **La clasificación binaria es más fácil que el problema real** porque se descartó la gama media. Para una versión más exigente se puede cambiar `MAPA_BIN` y dejar la gama media en una de las dos clases.
- **En GPU hay más filas de gama alta** (cada chipset potente tiene muchas variantes comerciales), por lo que Accuracy debe leerse junto con el F1 macro y el modelo base.
- **Las gamas son relativas a la muestra** que logró cruzarse entre catálogo y benchmark; no son un estándar del mercado, y se pierden los modelos cuyos nombres no coinciden.
- **La regresión predice el mismo puntaje que define las gamas:** mide qué tan bien se aproxima con las especificaciones, no el rendimiento real en un equipo ni los cuellos de botella entre CPU y GPU.

## Fuentes y créditos

- Catálogos de GPU, CPU y coolers: [PC Part Dataset](https://github.com/docyx/pc-part-dataset) (`docyx`).
- Benchmark de GPU (G3D Mark): [pc-game-fps-dataset](https://github.com/emilioagustin83/pc-game-fps-dataset) (`emilioagustin83`).
- Benchmark de CPU (CPU Mark): [cpu-benchmark-v9](https://github.com/spapas/cpu-benchmark-v9) (`spapas`).
- Mediciones de coolers (THETA): [Cybenetics](https://www.cybenetics.com/). Los datos pertenecen a sus autores originales.

## Autores

**David Cediel** y **David Gómez**
