# <center>**Proyecto final: clasificación de CPU, GPU y coolers**</center>

**Integrantes:** David Cediel y David Gómez

Proyecto de aprendizaje de máquina para clasificar procesadores (CPU), tarjetas gráficas (GPU) y coolers en **gamas de rendimiento**. Para CPU y GPU la gama se construye con puntuaciones de benchmarks y para los coolers con las mediciones de Cybenetics.

Este cuaderno reúne los **tres avances** del proyecto, separados en este orden:

| Avance | Contenido | Secciones |
|---|---|---|
| **Avance 1** | Construcción de los datasets y análisis exploratorio (EDA) | 1, 2 y 3 |
| **Avance 2** | Aprendizaje supervisado con cinco gamas: Fase 1 (hiperparámetros), Fase 2 (validación cruzada), Fase 3 (gráficas) y test final | 4 a 8 |
| **Avance 3** | **Clasificación binaria (gama baja vs. gama alta)**, **regresión** del puntaje del benchmark y elección de los **2 mejores modelos de clasificación y los 2 mejores de regresión** | 9, 10 y 11 |

> **Para ejecutarlo en Colab:** *Entorno de ejecución → Ejecutar todo*. Cuando lo pida, sube los dos CSV de coolers (`coolers_amd_am5.csv` y `coolers_intel_lga1851.csv`). Si ya están en la carpeta de Colab, no vuelve a pedirlos.

# AVANCE 1 — Datos y análisis exploratorio (EDA)

Este avance cubre la **construcción de los datasets**, la definición de las gamas y el **análisis descriptivo**. El entrenamiento de modelos empieza en el Avance 2: así se evita mezclar decisiones de exploración con la evaluación de modelos.

Como la variable a predecir tiene **cinco clases** (baja, media-baja, media, media-alta y alta), en los avances 1 y 2 el problema es de **clasificación multiclase**; en el Avance 3 se simplifica a dos clases y se agrega la regresión.

# **1. Librerías y funciones para preparar los datos**

Se cargan las librerías y se definen unas funciones sencillas para limpiar los nombres de los componentes y poder relacionarlos con sus benchmarks.

# **2. Construcción de los datasets**

## 2.1 Catálogos de componentes

Los catálogos traen precios y características de GPU, CPU y coolers. Se cargan desde GitHub y se revisan algunas filas.

## 2.2 Benchmarks de CPU y GPU

Para las GPU se usa **G3D Mark** y para las CPU **CPU Mark**. Se limpian los nombres para poder relacionar cada componente con su puntuación.

## 2.3 Mediciones de los coolers

Aquí se suben los CSV de Cybenetics (`coolers_amd_am5.csv` y `coolers_intel_lga1851.csv`). Si un mismo modelo aparece en los dos archivos, se juntan sus datos.

## 2.4 Gamas de GPU y CPU

Las gamas se crean con la puntuación del benchmark: se ordenan los modelos y se dividen en **cinco grupos del mismo tamaño**. En las GPU se conserva cada variante comercial del catálogo (por eso varias tarjetas pueden compartir chipset y gama).

## 2.5 Gamas de coolers

Los coolers se clasifican con la columna **Overall performance** de Cybenetics. Como AMD e Intel se miden por separado, primero se calcula el percentil dentro de cada plataforma y después se promedia.

# **3. Exploración de datos**

Se revisa cuántos productos hay en cada gama y cómo cambian algunas características de una gama a otra.

## 3.1 Contexto, justificación y decisiones de diseño

**Problema.** Clasificar CPU, GPU y coolers en cinco gamas de rendimiento a partir de atributos disponibles en los catálogos. Esto permite explorar si las características técnicas y comerciales permiten aproximar una gama sin consultar directamente el benchmark que se utilizó para construir la etiqueta.

**Origen de los datos.** Los catálogos de CPU, GPU y coolers se descargan de PC Part Dataset; las puntuaciones de CPU y GPU provienen de sus fuentes de benchmark respectivas; los coolers se evalúan con mediciones de Cybenetics cargadas por el usuario. El proceso de unión usa claves normalizadas y puede perder modelos cuyos nombres no se logran relacionar.

**¿Por qué aprendizaje automático?** Una vez creada una etiqueta de cinco clases, los modelos supervisados permiten comparar reglas lineales y no lineales, medir generalización en datos no usados para ajustar el modelo y cuantificar los errores entre gamas. El resultado no reemplaza el benchmark: aprende a aproximar la etiqueta construida.

| Decisión de diseño | Decisión aplicada | Justificación / precaución |
|---|---|---|
| Unidad de análisis | Producto/modelo relacionado con catálogo y benchmark | Las variantes comerciales pueden repetirse; se controlan grupos por `match_key` cuando se parte el dataset. |
| Variable objetivo | `gama` (cinco clases ordenadas) | Se deriva del rendimiento/percentil, no se introduce como predictor. |
| Etiquetas | Cinco grupos de tamaño aproximadamente similar | Los límites son relativos a la muestra disponible, no estándares universales del mercado. |
| Predictores | Características de catálogo; variables categóricas codificadas | Se excluyen etiqueta, identificadores y columnas que revelan directamente el benchmark/percentil. |
| Partición | 60% train, 20% validación, 20% test | Test se reserva para una evaluación final. Se agrupa por clave para evitar que un mismo modelo quede en particiones distintas. |
| Selección | F1 macro en validación y validación cruzada | Da el mismo peso a cada gama y reduce el sesgo de accuracy cuando hay diferencias de frecuencias. |
| Datos faltantes | Imputación con mediana dentro del pipeline | La mediana se aprende en entrenamiento, evitando estimar imputación con validación/test. |
| Escalamiento | StandardScaler dentro del pipeline SVM | Las distancias del SVM dependen de la escala de las variables. |

## 3.2 Calidad y estructura de los datasets

Las tablas siguientes muestran dimensiones, tipos, duplicados y valores faltantes antes del modelado. Los faltantes no se rellenan globalmente en el EDA: se documentan aquí y la imputación de predictores se ajusta dentro de cada pipeline de entrenamiento.

## 3.3 Distribución de la variable objetivo

La clasificación por cuantiles intenta equilibrar las cinco gamas según la puntuación usada para construirlas. Sin embargo, la distribución por filas puede variar si hay varias variantes de catálogo que comparten una misma clave.

## 3.4 Histogramas, boxplots y tabla resumen numérica

Se revisan todas las variables numéricas que tengan variación. La tabla incluye media, mediana, desviación estándar, asimetría y rango intercuartílico (IQR). **Regla descriptiva:** si la asimetría absoluta es menor que 1 y media/mediana son cercanas, la media puede resumir razonablemente el centro; si hay asimetría marcada o valores extremos, se prefiere la mediana. Es una regla orientativa, no una prueba estadística.

## 3.5 Correlación entre variables numéricas

La correlación de Pearson describe asociaciones lineales entre variables numéricas; no implica causalidad. Las columnas que construyen la etiqueta se muestran separadas de la matriz de predictores para no confundir una asociación esperable con capacidad predictiva legítima.

## 3.6 Nulos y valores atípicos: criterios de manejo

- **Nulos:** en el EDA se cuantifican, no se reemplazan por cero por defecto porque cero puede ser un valor real (por ejemplo, ruido no reportado no equivale a ruido cero). Durante entrenamiento, las variables numéricas se imputan con la mediana calculada solo sobre el fold de entrenamiento; las categorías faltantes se tratan según la preparación de variables del pipeline.
- **Outliers:** los boxplots señalan observaciones fuera de 1.5×IQR, pero no prueban que sean errores. En componentes de PC, precios, consumo, dimensiones o RPM pueden tener distribuciones genuinamente asimétricas. Por eso no se eliminan automáticamente; se conservan salvo que se identifique un error de calidad verificable.
- **Limitación:** las etiquetas de gama se derivan de benchmarks/percentiles; el EDA describe la muestra que logró cruzarse, no todos los productos del mercado.

## 3.7 Cierre del Avance 1

El Avance 1 deja documentados el origen de los datos, la construcción de las etiquetas, la calidad de los datos, la distribución de clases, las variables numéricas, sus valores extremos y sus correlaciones. El **Avance 2** comienza con la preparación para aprendizaje supervisado, la evaluación de fuga de información, el entrenamiento de GNB, árbol, Random Forest y SVM, la validación cruzada y el test final.

# AVANCE 2 — Aprendizaje supervisado (cinco gamas)

Este avance parte de los datasets del Avance 1. Se prepara la partición Train / Validación / Test, se mide la fuga de información y se entrenan **GNB, árbol de decisión, Random Forest y SVM** en tres fases: **Fase 1** (hiperparámetros en validación), **Fase 2** (validación cruzada) y **Fase 3** (gráficas). Termina con la evaluación final en test y las conclusiones.

# **4. X, y y partición Train / Validación / Test**

- Se reserva el **20% para test** y se consulta únicamente al final.
- El 80% restante se divide en **train (60% del total)** y **validación (20% del total)**.
- Se entrena con `X_train, y_train` y se comparan las configuraciones con `X_val, y_val`.

**La respuesta (`y`) es la gama.** Para que el modelo no vea la respuesta, no se usan como entrada las puntuaciones con las que se crearon las gamas (`benchmark_score`, `score_percentile`), ni los nombres de los productos.

Las variables de texto (color, tipo de cooler, microarquitectura, etc.) se convierten en columnas de 0 y 1 con `pd.get_dummies`. Los valores vacíos se completan con la **mediana dentro del `Pipeline`**, así que se calculan solo con los datos de entrenamiento.

En las GPU hay varias tarjetas del mismo chipset (tienen la misma gama y casi las mismas características). Para que un mismo chipset no quede a la vez en train y en test, la partición se hace **por chipset** (`match_key`), manteniendo la proporción de gamas.

## 4.1 Medición de fuga de información (antes del entrenamiento)

Se comparan tres escenarios sobre la misma partición train/validación usando Random Forest como diagnóstico, no como selección definitiva. A: variables permitidas. B: variables permitidas + puntuaciones/percentiles que ayudaron a construir la etiqueta. C: además identificadores textuales del producto codificados. Si B o C mejora mucho, puede indicar que el modelo está recuperando información que define o identifica la etiqueta, no aprendiendo características generalizables. Los escenarios B/C son únicamente diagnósticos y no se usan para elegir el modelo final.

# **5. Fase 1: GNB, DT, RF y SVM con distintos hiperparámetros**

Se entrenan los cuatro modelos con `X_train` y se evalúan en `X_val`. Para cada modelo se varían sus hiperparámetros principales:

| Modelo | Hiperparámetros que se varían |
|---|---|
| **GNB** (Gaussian Naive Bayes) | `var_smoothing` |
| **DT** (árbol de decisión) | `max_depth` y `min_samples_leaf` |
| **RF** (Random Forest) | `n_estimators` y `max_depth` |
| **SVM** | `kernel` (linear / rbf) y `C` |

También se incluye un **modelo base** que siempre responde la gama más frecuente, para saber si los modelos aportan algo.

Todos los modelos usan el mismo `Pipeline`: primero se completan los vacíos con la mediana y, en el SVM, se estandarizan las variables (como en el notebook 07). El criterio para elegir es el **F1 macro** en validación; si hay empate, se conserva el primero de la tabla.

### Conclusiones de la Fase 1

- **GPU:** Random Forest (`n_estimators=200`, `max_depth=8`, `max_features='sqrt'`, `class_weight='balanced'`) obtuvo **F1=0,671** (`F1 train=0,873`); SVM RBF (`C=1`, `gamma=0,1`) obtuvo **F1=0,644** (`F1 train=0,751`). Ambos superaron al modelo base (**F1=0,085**). Las brechas train-validación fueron 0,202 y 0,107, respectivamente; RF muestra mayor riesgo de sobreajuste.

- **CPU:** Random Forest (`n_estimators=100`, sin límite explícito de profundidad, `max_features='sqrt'`) obtuvo **F1=0,801** (`F1 train=1,000`) y SVM lineal (`C=10`) **F1=0,800** (`F1 train=0,936`). Ambos superaron al modelo base (**F1=0,066**). La diferencia entre ellos es mínima (0,001); SVM tiene una brecha menor (0,136 frente a 0,199), mientras que RF presenta más señales de sobreajuste.

- **Coolers:** SVM RBF (`C=1`, `gamma=0,01`, `class_weight='balanced'`) obtuvo **F1=0,467** (`F1 train=0,702`) y el Árbol de Decisión (`criterion='gini'`, `max_depth=2`, `min_samples_leaf=1`, clases balanceadas) **F1=0,313** (`F1 train=0,580`). Superaron al modelo base (**F1=0,067**), con brechas de 0,235 y 0,267. Dado que solo hay 48 modelos, estos resultados son preliminares y pueden variar con otra partición.

En los tres datasets, los modelos seleccionados superaron al modelo base en el F1 macro de validación. Las configuraciones candidatas pasan a la Fase 2 para comparar la media y la variabilidad entre folds antes de escoger el modelo final.

# **6. Fase 2: validación cruzada K-Fold con los dos mejores candidatos de cada dataset**

La validación cruzada permite ver cuánto cambian los resultados entre particiones. Como en el notebook 07, se hace **solo sobre `X_train, y_train`**: validación y test quedan fuera de los folds.

Para tener **distintas particiones** se prueba con **K = 3 y K = 5** y con dos semillas distintas para mezclar los datos (4 particiones en total). Los folds son estratificados (en las GPU, además, un mismo chipset no queda en dos folds). La media resume el desempeño y la desviación estándar describe la variación entre folds; no es un intervalo de confianza.

### 6.1 Resultados fold por fold y estabilidad de métricas

Se conserva cada fold para poder revisar variabilidad y no solo el promedio. Los boxplots resumen Accuracy, Precision macro, Recall macro y F1 macro de los cuatro algoritmos seleccionados en las diez repeticiones. Los folds no son observaciones independientes perfectas; se interpretan como evidencia de estabilidad, no como intervalos de confianza formales.

### 6.2 Sensibilidad, falsos positivos y brecha train-validación

En un problema multiclase se calcula especificidad como promedio one-vs-rest por clase. Para cada clase, TN/FP/FN/TP se obtienen de su matriz binaria frente al resto. La tasa de falsas alarmas por cada 100 productos se calcula como FP/(FP+TN)×100. La brecha train-validación compara F1 macro de entrenamiento y validación; una brecha positiva grande puede sugerir sobreajuste.

# **7. Fase 3: gráficas**

## 7.1 Comparación de métricas

Se comparan Accuracy, Precision, Recall y F1 de la mejor configuración de cada modelo en validación, junto con el modelo base.

## 7.2 Matrices de confusión por modelo

Muestran en qué gamas se equivoca cada modelo (mejor configuración de cada uno, entrenada con `X_train` y evaluada en validación).

## 7.3 Impacto de los hiperparámetros

Cada gráfica muestra cómo cambia el F1 macro de validación al variar los hiperparámetros de un modelo.

### 7.5 Visualización de un árbol de decisión

Se muestra un árbol limitado a tres niveles para facilitar su lectura. Los nombres de variables se obtienen del pipeline después de la imputación; las reglas son asociaciones aprendidas en esta muestra, no relaciones causales.

# **8. Evaluación final en test**

Para cada dataset se toma el modelo elegido en la Fase 2, se **reentrena con train + validación (80% del total)** y se evalúa **una sola vez** en test. Como en el notebook 07, el test no se usa para cambiar ninguna decisión.

## Conclusiones y limitaciones del Avance 2

Los resultados permiten comparar los modelos para CPU, GPU y coolers. Las gamas dependen de los productos que lograron relacionarse con los benchmarks, y algunos pueden quedar por fuera si sus nombres no coinciden. En GPU se separan los datos por chipset para evitar que el mismo chipset quede en entrenamiento y prueba. El resultado de los coolers depende de los CSV de Cybenetics que se carguen y, como hay pocos modelos (menos de 50), una sola equivocación mueve mucho las métricas de validación y de test.

La clasificación usa las características disponibles, pero no garantiza el rendimiento en cualquier equipo. Tampoco calcula cuellos de botella entre CPU y GPU, porque para eso harían falta datos de juegos y de combinaciones reales.

# AVANCE 3 — Clasificación binaria y regresión

En los avances anteriores se clasificó cada componente en **cinco gamas**. En este avance se hacen dos cosas nuevas con los mismos tres datasets (GPU, CPU y coolers):

| Parte | Qué se hace | Resultado que se busca |
|---|---|---|
| **A. Clasificación binaria** | La gama se simplifica a **dos clases: gama baja vs. gama alta** y se prueba el modelo | Los **2 mejores modelos de clasificación** |
| **B. Regresión** | En vez de la gama, se predice el **puntaje continuo** del benchmark con el que se construyó la gama | Los **2 mejores modelos de regresión** |

Se mantienen las mismas particiones Train / Validación / Test del Avance 2 (por `match_key`), así los resultados se pueden comparar con justicia. El test se usa **una sola vez al final** de cada parte y no se usa para elegir nada.

# **9. Parte A: Clasificación binaria (gama baja vs. gama alta)**

## 9.1 Cómo se pasa de 5 gamas a 2 clases

Se agrupan las gamas de los extremos y se **descarta la gama media**, porque sus productos están justo en el centro y no pertenecen claramente a ninguno de los dos lados:

| Gama original | Clase binaria |
|---|---|
| Gama baja, Gama media-baja | **Gama baja** |
| Gama media | *(se descarta)* |
| Gama media-alta, Gama alta | **Gama alta** |

La clase **positiva** es *Gama alta*. Precision, Recall y F1 se siguen calculando con promedio **macro** (las dos clases pesan igual). En **CPU y coolers** quedan clases de igual tamaño (50% / 50%). En **GPU** la gama alta tiene más filas (entre 62% y 72% según el conjunto) porque los chipsets más potentes tienen muchas variantes comerciales; por eso en GPU el modelo base ya acierta cerca del 65% y hay que mirar F1 macro, no solo Accuracy.

Los productos de test siguen siendo los mismos del Avance 2 (menos los de gama media), así que no hay mezcla entre train y test.

## 9.2 Fase 1 (binaria): los mismos cuatro algoritmos con distintos hiperparámetros

Se reutilizan las **mismas 170 configuraciones** del Avance 2 (GNB, DT, RF, SVM y el modelo base). Se entrena con `X_train` y se evalúa en `X_val`. El criterio para elegir es el **F1 macro** en validación.

## 9.3 Fase 2 (binaria): validación cruzada K-Fold

Se pasa a la validación cruzada la **mejor configuración de cada uno de los cuatro algoritmos** (sin el modelo base). Igual que en el Avance 2: solo sobre `X_train`, K = 5 (o menos si alguna clase tiene pocos grupos), **10 repeticiones** con semillas distintas, folds estratificados y agrupados por `match_key`. Al final se eligen los **2 mejores modelos de clasificación** por F1 macro medio.

## 9.4 Probar el modelo: evaluación final en test (binaria)

Los **2 mejores modelos** de cada dataset se **reentrenan con train + validación** y se evalúan **una sola vez** en test. Se comparan con el modelo base (que siempre responde la clase más frecuente). Además de Accuracy, Precision, Recall y F1 se calcula la **especificidad** y el **AUC-ROC**.

## 9.5 Lectura de los resultados: ¿por qué la clasificación binaria sale tan alta?

Los resultados de la binaria son mucho más altos que los de cinco gamas (casi 1.0). Es esperable: **se quitó la gama media**, que es justo donde un producto se parece a sus vecinos de arriba y de abajo. Lo que queda son los extremos, que se separan con reglas simples. No es fuga de información (el corte usa características reales del catálogo, no el benchmark), pero **sí hace que el resultado sea optimista**. Las dos celdas siguientes lo comprueban: (1) cuánto logra un árbol de **un solo corte** y con qué variable, y (2) qué hace el mejor modelo con los productos de **gama media** de test, que nunca vio.

# **10. Parte B: Regresión (predecir el puntaje del benchmark)**

## 10.1 Variable objetivo continua

La gama se construyó a partir de un **puntaje numérico**. En la regresión se predice ese puntaje directamente en vez de la categoría:

| Dataset | Variable a predecir (`y`) | Qué es |
|---|---|---|
| GPU | `benchmark_score` | Puntaje **G3D Mark** |
| CPU | `benchmark_score` | Puntaje **CPU Mark** |
| COOLER | `score_percentile` | Percentil de rendimiento (0 a 1) dentro de su plataforma |

Las variables de entrada (`X`) son **las mismas del Avance 2**: no incluyen el puntaje, el percentil, la gama ni los nombres, para evitar fuga de información. Se mantienen las mismas particiones Train / Validación / Test.

Todos los modelos se entrenan dentro de un `TransformedTargetRegressor` que estandariza `y` (así el SVR no se ve afectado por la escala de los puntajes) y las métricas se calculan en las unidades originales. Se usan **MAE**, **RMSE** y **R²**; el modelo base siempre predice el promedio del entrenamiento (su R² es cercano a 0).

## 10.2 Fase 1 (regresión): cuatro algoritmos con distintos hiperparámetros

Los cuatro algoritmos son el análogo de regresión de los del Avance 2. El modelo lineal es el punto de partida (*baseline*):

| Modelo | Hiperparámetros que se varían |
|---|---|
| **Lineal** | Regresión lineal simple y **Ridge** con `alpha` = 1, 10, 100 |
| **DT** (árbol de regresión) | `max_depth` y `min_samples_leaf` |
| **RF** (Random Forest) | `n_estimators`, `max_depth` y `max_features` |
| **SVR** (kernel rbf) | `C` y `epsilon` |

El criterio para elegir es el **R²** en validación (si hay empate, el menor MAE).

## 10.3 Fase 2 (regresión): validación cruzada y elección de los 2 mejores

Igual que en clasificación, la mejor configuración de cada algoritmo (sin el modelo base) pasa a validación cruzada sobre `X_train`: K = 5, **10 repeticiones**, folds agrupados por `match_key` y estratificados por la gama original. Los **2 mejores regresores** son los de mayor **R² medio**.

## 10.4 Probar los 2 mejores regresores: evaluación final en test

Cada regresor elegido se **reentrena con train + validación** y se evalúa **una sola vez** en test. Las gráficas muestran el valor real contra el predicho (la línea roja es la predicción perfecta) y la distribución de los residuos (real − predicho).

# **11. Resumen del Avance 3: los 2 mejores modelos de clasificación y de regresión**

# **Conclusiones del Avance 3**

> Los números de esta sección salen de la ejecución de este cuaderno (semillas fijas). Si se ejecuta con otra versión de scikit-learn pueden cambiar en los decimales. Los puestos se asignan con la **validación cruzada**; el test solo confirma y no se usó para elegir.

## A. Clasificación binaria (gama baja vs. gama alta): los 2 mejores modelos


- **Todos superan al modelo base** (F1 en test: GPU 0.392, CPU 0.331, cooler 0.333).
- **Los dos primeros de cada dataset están empatados técnicamente:** la diferencia entre sus medias es menor que la desviación entre folds. El puesto se decide por la media y, si empatan, por la menor desviación.
- **¿Por qué es tan alto frente a las cinco gamas?** En el Avance 2 el F1 en test fue 0,683 (GPU), 0,821 (CPU) y 0,167 (cooler). Al quitar la gama media quedan solo los extremos, que se separan con reglas simples: un árbol de **un solo corte** obtiene 100% de acierto en GPU (`memory` ≤ 5 GB), 87% en CPU (`core_count` ≤ 3.5) y 100% en cooler (líquido vs. aire). No es fuga de información (son características reales del catálogo), pero el desempeño es **optimista**.
- **Gama media (no vista):** el mejor modelo manda 68% de las 105 GPU de gama media a *gama baja*; en CPU manda 68% de 25 a *gama alta*; en cooler solo hay 2 productos de gama media en test, así que no hay conclusión.

## B. Regresión (predecir el puntaje del benchmark): los 2 mejores modelos


- **Todos superan al modelo base** (R² en test: GPU -0.014, CPU -0.001, cooler -0.007; predecir el promedio da R² ≈ 0).
- **El modelo lineal con regularización (Ridge) es competitivo:** queda entre los 2 mejores en GPU y CPU. La relación entre las especificaciones y el puntaje es en buena parte monótona; en GPU el Random Forest lo supera por menos que la desviación entre folds.
- En CPU, el segundo puesto tiene mejor R² en test que el primero, pero la diferencia en validación cruzada era menor que la desviación; **no se cambió la elección tras ver el test.**
- **Variables más importantes** (Random Forest): GPU `memory`, `core_clock` y `boost_clock`; CPU `core_count`, `tdp` y `boost_clock`; cooler `peso_g`, tipo de refrigeración y número de ventiladores.

## C. Resumen: los modelos elegidos para el proyecto

- **Clasificación (2 mejores):** una SVM lineal y un árbol de decisión en GPU y CPU; un árbol de un nivel y un Random Forest en cooler.
- **Regresión (2 mejores):** Random Forest y Ridge en GPU; Ridge y Random Forest en CPU; árbol de profundidad 2 y Random Forest en cooler.

## D. Limitaciones

- **Cooler tiene muy pocos datos:** 48 modelos en total (22 / 8 / 8 en train / validación / test para la binaria y 28 / 10 / 10 en regresión). Un solo producto mueve mucho las métricas; sus resultados son preliminares.
- **La binaria es más fácil que el problema real** porque se descartó la gama media. Para una versión más exigente se puede cambiar `MAPA_BIN` y dejar la gama media en una de las dos clases.
- **En GPU hay más filas de gama alta** (cada chipset potente tiene muchas variantes comerciales), por lo que Accuracy debe leerse junto con F1 macro y con el modelo base.
- **La regresión predice el mismo puntaje que define las gamas;** mide qué tan bien se aproxima ese puntaje con las especificaciones del catálogo, no el rendimiento real en cualquier equipo.
