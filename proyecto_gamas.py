"""
Proyecto final: clasificación de CPU, GPU y coolers en gamas de rendimiento
Autores: David Cediel y David Gómez

Este archivo contiene SOLO el código de notebooks/Avance_3_ML_y_Terminado.ipynb
(Avance 1, Avance 2 y Avance 3), en el mismo orden. El texto explicativo está en
README.md y en docs/informe.md.

Cómo ejecutarlo (desde la carpeta raíz del repositorio):
    python src/proyecto_gamas.py

- Descarga de internet los catálogos de GPU, CPU y coolers y los benchmarks de CPU y GPU.
- Lee los dos CSV de Cybenetics que están en la carpeta data/.
- Las tablas se imprimen en la consola y las gráficas se guardan en outputs/figuras/.
- También se puede abrir en VS Code o Spyder: los marcadores "# %%" separan las celdas.
"""




# %% Preparación para ejecutar como script (rutas, gráficas y tablas)
from pathlib import Path
import matplotlib.pyplot as plt
import IPython.display as _ipd

try:
    RAIZ_PROYECTO = Path(__file__).resolve().parent.parent   # carpeta raíz del repositorio
except NameError:                                            # en Colab / Jupyter no existe __file__
    RAIZ_PROYECTO = Path.cwd()
CARPETA_DATOS = RAIZ_PROYECTO / "data"

try:
    from IPython import get_ipython
    EN_NOTEBOOK = get_ipython() is not None
except ImportError:
    EN_NOTEBOOK = False

if not EN_NOTEBOOK:
    # Como script no hay dónde mostrar las gráficas: se guardan como PNG en outputs/figuras/
    plt.switch_backend("Agg")
    CARPETA_FIGURAS = RAIZ_PROYECTO / "outputs" / "figuras"
    CARPETA_FIGURAS.mkdir(parents=True, exist_ok=True)
    _contador_figuras = [0]

    def _guardar_figura(*args, **kwargs):
        _contador_figuras[0] += 1
        plt.savefig(CARPETA_FIGURAS / f"figura_{_contador_figuras[0]:03d}.png", dpi=110, bbox_inches="tight")
        plt.close("all")

    def _mostrar(*objetos, **kwargs):
        # display() de Jupyter reemplazado por una impresión de texto en la consola
        for obj in objetos:
            if obj.__class__.__name__ == "Markdown":
                print(obj.data)
            elif hasattr(obj, "to_string"):
                print(obj.to_string())
            else:
                print(obj)

    plt.show = _guardar_figura
    _ipd.display = _mostrar


###############################################################################
# AVANCE 1 — Datos y análisis exploratorio (EDA)
###############################################################################

###############################################################################
# 1. Librerías y funciones para preparar los datos
###############################################################################

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import io
import re
import time
import unicodedata
from IPython.display import display
try:
    from google.colab import files
except ImportError:      # fuera de Colab no existe; se leen los CSV de la carpeta
    files = None

from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedGroupKFold, cross_validate
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, confusion_matrix,
                             f1_score, classification_report, ConfusionMatrixDisplay,
                             make_scorer)

sns.set_theme()
SEED = 21
GAMAS = ["Gama baja", "Gama media-baja", "Gama media", "Gama media-alta", "Gama alta"]

# %% funciones para limpiar nombres y convertir valores a números
def normalizar_columnas(df):
    # Deja los nombres de las columnas en minúscula y con guion bajo
    df = df.copy()
    nombres = []
    for columna in df.columns:
        nombre = str(columna).lower()
        nombre = re.sub(r"[^a-z0-9]+", "_", nombre).strip("_")
        nombres.append(nombre)
    df.columns = nombres
    return df


def convertir_numero(valor):
    # Convierte textos en un número
    if pd.isna(valor):
        return np.nan
    texto = str(valor).strip()
    texto = texto.replace("$", "").replace("€", "")
    # Algunos precios usan coma para separar los miles
    if re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", texto):
        texto = texto.replace(",", "")
    partes_numero = re.findall(r"\d+(?:\.\d+)?", texto)
    if len(partes_numero) == 0:
        return np.nan
    numeros = []
    for parte in partes_numero:
        numeros.append(float(parte))
    return max(numeros)


def normalizar_texto(valor):
    # Minúscula, sin tildes y sin espacios repetidos
    if pd.isna(valor):
        return ""
    texto = str(valor).lower()
    texto = unicodedata.normalize("NFKD", texto)
    texto = texto.encode("ascii", "ignore").decode("utf-8")
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()

# %% funciones para relacionar CPU, GPU y coolers con sus benchmarks
def clave_gpu(valor):
    texto = normalizar_texto(valor)
    texto = re.sub(r"\b(nvidia|geforce|amd|radeon|intel)\b", " ", texto)
    texto = re.sub(r"\b\d+\s*(gb|gib)\b", " ", texto)
    texto = re.sub(r"[^a-z0-9]+", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def clave_cpu(valor):
    texto = normalizar_texto(valor)
    texto = re.sub(r"\s*@?\s*\d+(?:\.\d+)?\s*ghz\b", " ", texto)
    texto = re.sub(r"\b\d+(?:\.\d+)?\s*ghz\b", " ", texto)
    texto = re.sub(r"\bwith\s+radeon\s+graphics\b", " ", texto)
    texto = re.sub(r"\b\d+\s*[- ]?\s*cores?\b", " ", texto)
    texto = re.sub(r"\b\d+\s*[- ]?\s*threads?\b", " ", texto)
    texto = re.sub(r"\b(processor|cpu|apu)\b", " ", texto)
    texto = re.sub(r"[^a-z0-9]+", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


MARCAS_COOLER = [
    "cooler master", "thermalright", "deepcool", "arctic", "noctua",
    "corsair", "be quiet", "thermaltake", "nzxt", "scythe", "enermax",
    "lian li", "fractal design", "id cooling", "alphacool", "silverstone",
    "antec", "raijintek", "cryorig", "zalman", "montech", "amd", "intel"
]


def clave_cooler(fabricante, modelo):
    texto = normalizar_texto(f"{fabricante or ''} {modelo or ''}")
    texto = re.sub(r"\b(amd\s*am5|intel\s*lga\s*1851|lga1851|am5)\b", " ", texto)
    # Se quita la marca para poder comparar nombres parecidos
    for marca in sorted(MARCAS_COOLER, key=len, reverse=True):
        texto = re.sub(r"\b" + re.escape(marca) + r"\b", " ", texto)
    texto = re.sub(r"\b(theta|silver|gold|bronze|titanium|platinum|unrated)\b", " ", texto)
    texto = re.sub(r"[^a-z0-9]+", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()

# %% funciones para organizar los datos y asignar las gamas
def limpiar_catalogo(df):
    df = normalizar_columnas(df)
    df = df.drop_duplicates().copy()
    columnas_numericas = ["price", "memory", "core_clock", "boost_clock", "length",
                          "core_count", "tdp", "rpm", "noise_level", "size", "wattage"]
    for columna in columnas_numericas:
        if columna in df.columns:
            df[columna] = df[columna].apply(convertir_numero)
    return df


def primera_no_vacia(serie):
    valores = serie.dropna()
    valores = valores[valores.astype(str).str.strip().ne("")]
    if len(valores) == 0:
        return np.nan
    return valores.iloc[0]


def preparar_ranking(df, columna_modelo, columna_score, funcion_clave):
    ranking = df[[columna_modelo, columna_score]].copy()
    ranking["match_key"] = ranking[columna_modelo].apply(funcion_clave)
    ranking["benchmark_score"] = ranking[columna_score].apply(convertir_numero)
    ranking = ranking.dropna(subset=["match_key", "benchmark_score"])
    ranking = ranking[ranking["match_key"].ne("")].copy()
    ranking = ranking.groupby("match_key", as_index=False).agg(
        benchmark_model=(columna_modelo, "first"),
        benchmark_score=("benchmark_score", "median")
    )
    return ranking


def asignar_gamas(df, columna_score):
    # Cinco grupos del mismo tamaño según la puntuación (de menor a mayor)
    df = df.dropna(subset=[columna_score]).copy()
    posiciones = df[columna_score].rank(method="first")
    df["gama"] = pd.qcut(posiciones, q=5, labels=GAMAS).astype(str)
    return df


def consolidar_catalogo(df, columna_modelo, funcion_clave):
    # Si un modelo aparece varias veces se deja una sola fila (mediana en números)
    df = limpiar_catalogo(df.copy())
    df["match_key"] = df[columna_modelo].apply(funcion_clave)
    df = df[df["match_key"].ne("")].copy()
    identificadores = {"match_key", "name", "chipset", "manufacturer"}
    numericas = [c for c in df.select_dtypes(include=np.number).columns if c not in identificadores]
    categoricas = [c for c in df.columns if c not in identificadores and c not in numericas]
    agregaciones = {c: "median" for c in numericas}
    agregaciones.update({c: "first" for c in categoricas})
    agregaciones["name"] = "first"
    return df.groupby("match_key", as_index=False).agg(agregaciones)

###############################################################################
# 2. Construcción de los datasets
###############################################################################

# -----------------------------------------------------------------------------
# 2.1 Catálogos de componentes
# -----------------------------------------------------------------------------

# %% cargar los catálogos de GPU, CPU y coolers
URLS_COMPONENTES = {
    "GPU": "https://raw.githubusercontent.com/docyx/pc-part-dataset/main/data/csv/video-card.csv",
    "CPU": "https://raw.githubusercontent.com/docyx/pc-part-dataset/main/data/csv/cpu.csv",
    "COOLER": "https://raw.githubusercontent.com/docyx/pc-part-dataset/main/data/csv/cpu-cooler.csv"
}

datos_originales = {}
for tipo, url in URLS_COMPONENTES.items():
    respuesta = requests.get(url, timeout=30)
    respuesta.raise_for_status()
    tabla = limpiar_catalogo(pd.read_csv(io.BytesIO(respuesta.content)))
    datos_originales[tipo] = tabla
    print(f"{tipo}: {tabla.shape[0]} filas x {tabla.shape[1]} columnas")
    display(tabla.head(3))

# -----------------------------------------------------------------------------
# 2.2 Benchmarks de CPU y GPU
# -----------------------------------------------------------------------------

# %% cargar los benchmarks de GPU y CPU
URL_GPU_BENCHMARK = "https://raw.githubusercontent.com/emilioagustin83/pc-game-fps-dataset/main/data/gpus.csv"
URL_CPU_BENCHMARK = "https://raw.githubusercontent.com/spapas/cpu-benchmark-v9/master/cpus.json"

respuesta_gpu = requests.get(URL_GPU_BENCHMARK, timeout=30)
respuesta_gpu.raise_for_status()
gpu_benchmark = normalizar_columnas(pd.read_csv(io.BytesIO(respuesta_gpu.content)))

respuesta_cpu = requests.get(URL_CPU_BENCHMARK, timeout=30)
respuesta_cpu.raise_for_status()
cpu_benchmark = normalizar_columnas(pd.read_json(io.BytesIO(respuesta_cpu.content)))

gpu_benchmark["g3d_mark"] = pd.to_numeric(gpu_benchmark["g3d_mark"], errors="coerce")
cpu_benchmark["cpu_mark"] = cpu_benchmark["cpu_mark"].apply(convertir_numero)

# Se dejan las GPU de escritorio (Desktop/Mobile solo como respaldo)
gpu_benchmark["tipo_normalizado"] = gpu_benchmark["type"].astype(str).str.lower().str.replace(r"\s+", "", regex=True)
gpu_benchmark = gpu_benchmark[gpu_benchmark["tipo_normalizado"].isin(["desktop", "desktop/mobile"])].copy()
gpu_benchmark["prioridad_tipo"] = gpu_benchmark["tipo_normalizado"].map({"desktop": 0, "desktop/mobile": 1})

gpu_benchmark["match_key"] = gpu_benchmark["name"].apply(clave_gpu)
gpu_benchmark = gpu_benchmark.dropna(subset=["g3d_mark"])
gpu_benchmark = gpu_benchmark[gpu_benchmark["match_key"].ne("")]
# Una sola fila por clave: se prefiere el registro Desktop
gpu_benchmark = gpu_benchmark.sort_values("prioridad_tipo").drop_duplicates("match_key", keep="first")

print("GPU con benchmark:", len(gpu_benchmark))
print("CPU con benchmark:", len(cpu_benchmark))
display(gpu_benchmark.head(3))
display(cpu_benchmark.head(3))

# -----------------------------------------------------------------------------
# 2.3 Mediciones de los coolers
# -----------------------------------------------------------------------------

# %% subir y unir los CSV de Cybenetics
import glob, os

# Si los CSV ya están en la carpeta de trabajo (por ejemplo /content) se usan; si no, se piden.
rutas_cooler = sorted(set(glob.glob("coolers_*.csv") + glob.glob(os.path.join(CARPETA_DATOS, "coolers_*.csv"))))
archivos_cooler = {}
if len(rutas_cooler) > 0:
    for ruta in rutas_cooler:
        with open(ruta, "rb") as f:
            archivos_cooler[os.path.basename(ruta)] = f.read()
    print("CSV encontrados en la carpeta:", list(archivos_cooler))
else:
    print("Selecciona los CSV de Cybenetics (AMD AM5 e Intel LGA1851).")
    archivos_cooler = files.upload()

frames_cooler = []
for nombre_archivo, contenido in archivos_cooler.items():
    if nombre_archivo.lower().endswith(".csv"):
        frames_cooler.append(normalizar_columnas(pd.read_csv(io.BytesIO(contenido))))

cooler_union = pd.concat(frames_cooler, ignore_index=True, sort=False)
cooler_union["match_key"] = [clave_cooler(f, m) for f, m in zip(cooler_union["manufacturer"], cooler_union["model"])]
cooler_union = cooler_union[cooler_union["match_key"].ne("")].copy()

# Un modelo en dos archivos se junta tomando el primer valor no vacío de cada columna
cooler_raw = cooler_union.groupby("match_key", as_index=False).agg(primera_no_vacia)

print("Modelos únicos de cooler:", len(cooler_raw))
display(cooler_raw.head(3))

# -----------------------------------------------------------------------------
# 2.4 Gamas de GPU y CPU
# -----------------------------------------------------------------------------

# %% gamas de GPU
catalogo_gpu = limpiar_catalogo(datos_originales["GPU"]).copy()
catalogo_gpu["match_key"] = catalogo_gpu["chipset"].apply(clave_gpu)
catalogo_gpu = catalogo_gpu[catalogo_gpu["match_key"].ne("")].copy()

ranking_gpu = preparar_ranking(gpu_benchmark, "name", "g3d_mark", clave_gpu)
ranking_gpu = ranking_gpu[ranking_gpu["match_key"].isin(set(catalogo_gpu["match_key"]))].copy()
ranking_gpu = asignar_gamas(ranking_gpu, "benchmark_score")

# Cada variante comercial se conserva (no se agrupa por chipset)
gpu_clasificado = catalogo_gpu.merge(ranking_gpu[["match_key", "benchmark_model", "benchmark_score", "gama"]],
                                     on="match_key", how="inner").reset_index(drop=True)

print("Chipsets con benchmark:", ranking_gpu["match_key"].nunique(), "| Filas GPU:", len(gpu_clasificado))
display(gpu_clasificado["gama"].value_counts().reindex(GAMAS).to_frame("cantidad"))

# %% gamas de CPU
ranking_cpu = preparar_ranking(cpu_benchmark, "name", "cpu_mark", clave_cpu)
catalogo_cpu = consolidar_catalogo(datos_originales["CPU"], "name", clave_cpu)

ranking_cpu = ranking_cpu[ranking_cpu["match_key"].isin(set(catalogo_cpu["match_key"]))].copy()
ranking_cpu = asignar_gamas(ranking_cpu, "benchmark_score")

cpu_clasificado = catalogo_cpu.merge(ranking_cpu[["match_key", "benchmark_model", "benchmark_score", "gama"]],
                                     on="match_key", how="inner").reset_index(drop=True)

print("CPU con benchmark:", len(cpu_clasificado))
display(cpu_clasificado["gama"].value_counts().reindex(GAMAS).to_frame("cantidad"))

# -----------------------------------------------------------------------------
# 2.5 Gamas de coolers
# -----------------------------------------------------------------------------

# %% gamas de coolers
# Columnas de rendimiento (una por plataforma)
col_score_amd = [c for c in cooler_raw.columns if "overall_performance" in c and "amd_am5" in c][0]
col_score_intel = [c for c in cooler_raw.columns if "overall_performance" in c and "intel_lga1851" in c][0]

# Especificaciones del cooler que se usarán como variables de entrada
mapa_features = {"type": "tipo_refrigeracion",
                 "radiator_size_air_dimensions_mm": "dimensiones",
                 "weight_g": "peso_g",
                 "fans": "ventiladores",
                 "noise_at_full_speed_dba": "ruido_dba"}
features_disponibles = {o: d for o, d in mapa_features.items() if o in cooler_raw.columns}

# Percentil de cada cooler dentro de su plataforma
mediciones = []
for plataforma, columna_score in [("AMD AM5", col_score_amd), ("Intel LGA1851", col_score_intel)]:
    valores = cooler_raw[columna_score].apply(convertir_numero)
    mask = valores.notna() & (valores > 0)
    parte = cooler_raw.loc[mask, ["match_key"]].copy()
    parte["plataforma"] = plataforma
    parte["percentil_plataforma"] = valores.loc[mask].rank(pct=True).values
    mediciones.append(parte)
cooler_mediciones = pd.concat(mediciones, ignore_index=True)

cooler_scores = cooler_mediciones.groupby("match_key", as_index=False).agg(
    score_percentile=("percentil_plataforma", "mean"),
    n_plataformas=("plataforma", "nunique"))

# Especificaciones por cooler
cooler_features = cooler_raw[["match_key"] + list(features_disponibles.keys())].rename(columns=features_disponibles)
for columna in ["peso_g", "ventiladores", "ruido_dba"]:
    if columna in cooler_features.columns:
        cooler_features[columna] = cooler_features[columna].apply(convertir_numero)

# Precio y otras características del catálogo (cuando el modelo se logra relacionar)
catalogo_cooler = consolidar_catalogo(datos_originales["COOLER"], "name", lambda n: clave_cooler("", n))
catalogo_cooler = catalogo_cooler.drop(columns=["name"], errors="ignore")

cooler_clasificado = cooler_scores.merge(cooler_features, on="match_key", how="left")
cooler_clasificado = cooler_clasificado.merge(catalogo_cooler, on="match_key", how="left")
cooler_clasificado = asignar_gamas(cooler_clasificado, "score_percentile").reset_index(drop=True)

print("Coolers con benchmark:", len(cooler_clasificado))
display(cooler_clasificado["gama"].value_counts().reindex(GAMAS).to_frame("cantidad"))

###############################################################################
# 3. Exploración de datos
###############################################################################

# %% juntar los tres datasets
datasets_clasificacion = {"GPU": gpu_clasificado, "CPU": cpu_clasificado, "COOLER": cooler_clasificado}

for nombre, df in datasets_clasificacion.items():
    print(f"{nombre}: {df.shape[0]} filas x {df.shape[1]} columnas")
    display(df.isna().sum().to_frame("valores vacíos").T)

# %% distribución de las gamas
plt.figure(figsize=(15, 4))
for i, (nombre, df) in enumerate(datasets_clasificacion.items()):
    plt.subplot(1, 3, i + 1)
    sns.countplot(data=df, x="gama", order=GAMAS)
    plt.title(f"Gamas: {nombre}")
    plt.xlabel("")
    plt.xticks(rotation=35)
plt.tight_layout()
plt.show()

# %% características numéricas por gama
columnas_score = ["benchmark_score", "score_percentile", "n_plataformas", "match_key"]

for nombre, df in datasets_clasificacion.items():
    numericas = [c for c in df.select_dtypes(include=np.number).columns if c not in columnas_score]
    numericas = [c for c in numericas if df[c].nunique() > 1][:4]
    plt.figure(figsize=(5 * len(numericas), 4))
    for i, col in enumerate(numericas):
        plt.subplot(1, len(numericas), i + 1)
        sns.boxplot(data=df, x="gama", y=col, order=GAMAS, showfliers=False)
        plt.xticks(rotation=40)
        plt.xlabel("")
        plt.title(col)
    plt.suptitle(f"Características por gama: {nombre}")
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------
# 3.2 Calidad y estructura de los datasets
# -----------------------------------------------------------------------------

# %%
for nombre, df in datasets_clasificacion.items():
    print(f"\n{'='*70}\n{nombre}: {df.shape[0]} filas × {df.shape[1]} columnas")
    print("Filas duplicadas:", int(df.duplicated().sum()))
    display(pd.DataFrame({
        "tipo": df.dtypes.astype(str),
        "no_nulos": df.notna().sum(),
        "nulos": df.isna().sum(),
        "% nulos": (100 * df.isna().mean()).round(2),
        "valores_distintos": df.nunique(dropna=True)
    }).sort_values("% nulos", ascending=False))

# -----------------------------------------------------------------------------
# 3.3 Distribución de la variable objetivo
# -----------------------------------------------------------------------------

# %%
for nombre, df in datasets_clasificacion.items():
    conteo = df["gama"].value_counts().reindex(GAMAS, fill_value=0)
    resumen = pd.DataFrame({"cantidad": conteo, "porcentaje": (100*conteo/conteo.sum()).round(2)})
    display(resumen.rename_axis("Gama").style.set_caption(f"Distribución de clases — {nombre}"))
    plt.figure(figsize=(7, 4))
    sns.countplot(data=df, x="gama", order=GAMAS)
    plt.title(f"Cantidad de productos por gama — {nombre}")
    plt.xticks(rotation=25)
    plt.xlabel("Gama"); plt.ylabel("Cantidad")
    plt.tight_layout(); plt.show()

# -----------------------------------------------------------------------------
# 3.4 Histogramas, boxplots y tabla resumen numérica
# -----------------------------------------------------------------------------

# %%
columnas_no_predictoras_eda = {"gama", "match_key", "name", "chipset", "benchmark_model",
                               "benchmark_score", "score_percentile", "n_plataformas"}
resumenes_eda = {}
for nombre, df in datasets_clasificacion.items():
    numericas = [c for c in df.select_dtypes(include=np.number).columns
                 if c not in columnas_no_predictoras_eda and df[c].nunique(dropna=True) > 1]
    if not numericas:
        print(f"{nombre}: no hay variables numéricas con variación para graficar.")
        continue
    filas = []
    for col in numericas:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        q1, q3 = s.quantile([.25, .75]) if len(s) else (np.nan, np.nan)
        asim = s.skew() if len(s) > 2 else np.nan
        filas.append({"variable": col, "n": len(s), "media": s.mean(), "mediana": s.median(),
                      "desv_std": s.std(), "min": s.min(), "Q1": q1, "Q3": q3, "max": s.max(),
                      "IQR": q3-q1, "asimetria": asim,
                      "resumen sugerido": ("mediana" if pd.notna(asim) and abs(asim) >= 1 else "media")})
    resumenes_eda[nombre] = pd.DataFrame(filas).set_index("variable")
    print(f"Resumen numérico — {nombre}")
    display(resumenes_eda[nombre].round(3))
    for col in numericas:
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
        sns.histplot(df[col].dropna(), kde=True, ax=ax[0])
        ax[0].set_title(f"Histograma: {col}")
        sns.boxplot(x=df[col], ax=ax[1])
        ax[1].set_title(f"Boxplot: {col}")
        plt.tight_layout(); plt.show()

# -----------------------------------------------------------------------------
# 3.5 Correlación entre variables numéricas
# -----------------------------------------------------------------------------

# %%
for nombre, df in datasets_clasificacion.items():
    numericas = [c for c in df.select_dtypes(include=np.number).columns
                 if c not in {"match_key"} and df[c].nunique(dropna=True) > 1]
    if len(numericas) < 2:
        print(f"{nombre}: no hay suficientes variables numéricas para una matriz de correlación.")
        continue
    corr = df[numericas].corr()
    plt.figure(figsize=(max(8, .7*len(numericas)), max(6, .6*len(numericas))))
    sns.heatmap(corr, cmap="coolwarm", center=0, vmin=-1, vmax=1)
    plt.title(f"Matriz de correlación numérica — {nombre}")
    plt.tight_layout(); plt.show()

###############################################################################
# AVANCE 2 — Aprendizaje supervisado (cinco gamas)
###############################################################################

###############################################################################
# 4. X, y y partición Train / Validación / Test
###############################################################################

# %% definir X, y y separar Train / Validación / Test
columnas_excluir = ["gama", "match_key", "name", "chipset", "benchmark_model",
                    "benchmark_score", "score_percentile", "n_plataformas"]

def separar_train_val_test(df, X, y):
    # Se separan las claves (match_key) para que un mismo chipset no quede en dos conjuntos
    claves = df[["match_key", "gama"]].drop_duplicates("match_key")
    claves_dev, claves_test = train_test_split(claves, test_size=0.20, stratify=claves["gama"], random_state=SEED)
    claves_train, claves_val = train_test_split(claves_dev, test_size=0.25, stratify=claves_dev["gama"], random_state=SEED)

    en_train = df["match_key"].isin(claves_train["match_key"])
    en_val = df["match_key"].isin(claves_val["match_key"])
    en_test = df["match_key"].isin(claves_test["match_key"])

    return {"X_train": X[en_train], "y_train": y[en_train],
            "X_val": X[en_val], "y_val": y[en_val],
            "X_test": X[en_test], "y_test": y[en_test],
            "X_dev": X[en_train | en_val], "y_dev": y[en_train | en_val],
            "grupos_train": df.loc[en_train, "match_key"],
            "df_test": df[en_test]}

datos = {}
for nombre, df in datasets_clasificacion.items():
    df = df.reset_index(drop=True)
    y = df["gama"]
    X = df.drop(columns=columnas_excluir, errors="ignore").dropna(axis=1, how="all")
    X = pd.get_dummies(X, dtype=float)
    datos[nombre] = separar_train_val_test(df, X, y)
    d = datos[nombre]
    print(f"{nombre}: {X.shape[1]} variables | Train: {len(d['y_train'])} | Validación: {len(d['y_val'])} | Test: {len(d['y_test'])}")
    display(d["y_train"].value_counts().reindex(GAMAS).rename("Train: muestras por gama").to_frame().T)

# -----------------------------------------------------------------------------
# 4.1 Medición de fuga de información (antes del entrenamiento)
# -----------------------------------------------------------------------------

# %%
filas_fuga = []
for nombre, d in datos.items():
    df0 = datasets_clasificacion[nombre].reset_index(drop=True)
    # Las máscaras se reconstruyen a partir de las posiciones guardadas en X; el split conserva los índices.
    idx_train, idx_val = d["X_train"].index, d["X_val"].index
    escenarios = {"A) Predictores permitidos": (d["X_train"].copy(), d["X_val"].copy())}
    columnas_proxy = [c for c in ["benchmark_score", "score_percentile", "n_plataformas"] if c in df0.columns]
    if columnas_proxy:
        xtr = d["X_train"].copy(); xva = d["X_val"].copy()
        for col in columnas_proxy:
            xtr[col] = df0.loc[idx_train, col].to_numpy()
            xva[col] = df0.loc[idx_val, col].to_numpy()
        escenarios["B) + variables proxy de la etiqueta"] = (xtr, xva)
    columnas_id = [c for c in ["name", "chipset", "benchmark_model"] if c in df0.columns]
    if columnas_id:
        idtr = pd.get_dummies(df0.loc[idx_train, columnas_id].fillna("Desconocido").astype(str), dtype=float)
        idva = pd.get_dummies(df0.loc[idx_val, columnas_id].fillna("Desconocido").astype(str), dtype=float)
        idtr, idva = idtr.align(idva, join="outer", axis=1, fill_value=0)
        xtr = d["X_train"].copy(); xva = d["X_val"].copy()
        xtr = pd.concat([xtr, idtr.set_axis(xtr.index)], axis=1)
        xva = pd.concat([xva, idva.set_axis(xva.index)], axis=1)
        escenarios["C) + identificadores textuales"] = (xtr, xva)
    for escenario, (xtr, xva) in escenarios.items():
        modelo_diag = make_pipeline(SimpleImputer(strategy="median"),
                                    RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                                           random_state=SEED, n_jobs=-1))
        modelo_diag.fit(xtr, d["y_train"])
        pred = modelo_diag.predict(xva)
        filas_fuga.append({"Dataset": nombre, "Escenario": escenario,
                           "F1 macro validación": f1_score(d["y_val"], pred, average="macro", zero_division=0)})
tabla_fuga = pd.DataFrame(filas_fuga)
display(tabla_fuga.pivot(index="Dataset", columns="Escenario", values="F1 macro validación").round(3))
print("Interpretación: un aumento grande al añadir proxies/identificadores alerta sobre fuga o memorización; la comparación no demuestra por sí sola causalidad.")

# %% some metrics
# Con cinco gamas se usa el promedio macro: todas las gamas pesan igual.
# zero_division=0 evita interrupciones si una configuración no predice alguna gama.
def metric_row(name, y_true, y_pred):
    return {"Modelo": name,
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "F1": f1_score(y_true, y_pred, average="macro", zero_division=0)}

scoring = {"Accuracy": "accuracy",
           "Precision": make_scorer(precision_score, average="macro", zero_division=0),
           "Recall": make_scorer(recall_score, average="macro", zero_division=0),
           "F1": make_scorer(f1_score, average="macro", zero_division=0)}

###############################################################################
# 5. Fase 1: GNB, DT, RF y SVM con distintos hiperparámetros
###############################################################################

# %% crear los modelos con sus hiperparámetros
# Cada candidato es: nombre -> (algoritmo, hiperparámetros, modelo)
def crear_candidatos():
    candidates = {}
    candidates["Base (gama más común)"] = ("Base", {}, make_pipeline(SimpleImputer(strategy="median"),
                                                                   DummyClassifier(strategy="most_frequent")))
    for vs in [1e-9, 1e-6, 1e-3]:
        candidates[f"GNB var_smoothing={vs}"] = ("GNB", {"var_smoothing": vs},
            make_pipeline(SimpleImputer(strategy="median"), GaussianNB(var_smoothing=vs)))

    for criterion in ["gini", "entropy", "log_loss"]:
        for depth in [1, 2, 4, 8, None]:
            for leaf in [1, 2, 4]:
                for weight in [None, "balanced"]:
                    candidates[f"DT {criterion} depth={depth} leaf={leaf} weight={weight}"] = (
                        "DT", {"criterion": criterion, "max_depth": str(depth),
                               "min_samples_leaf": leaf, "class_weight": str(weight)},
                        make_pipeline(SimpleImputer(strategy="median"),
                                      DecisionTreeClassifier(criterion=criterion, max_depth=depth,
                                          min_samples_leaf=leaf, class_weight=weight, random_state=SEED)))

    for n in [50, 100, 200]:
        for depth in [4, 8, None]:
            for max_features in ["sqrt", 0.5]:
                for weight in [None, "balanced"]:
                    candidates[f"RF n={n} depth={depth} features={max_features} weight={weight}"] = (
                        "RF", {"n_estimators": n, "max_depth": str(depth),
                               "max_features": str(max_features), "class_weight": str(weight)},
                        make_pipeline(SimpleImputer(strategy="median"),
                                      RandomForestClassifier(n_estimators=n, max_depth=depth,
                                          max_features=max_features, class_weight=weight,
                                          random_state=SEED, n_jobs=-1)))

    for kernel in ["linear", "rbf"]:
        for C in [0.1, 1, 10, 100]:
            for weight in [None, "balanced"]:
                gamma_values = ["scale"] if kernel == "linear" else ["scale", 0.01, 0.1, 1]
                for gamma in gamma_values:
                    candidates[f"SVM {kernel} C={C} gamma={gamma} weight={weight}"] = (
                        "SVM", {"kernel": kernel, "C": C, "gamma": str(gamma), "class_weight": str(weight)},
                        make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                      SVC(kernel=kernel, C=C, gamma=gamma, class_weight=weight)))
    return candidates

candidates = crear_candidatos()
print("Configuraciones por dataset:", len(candidates))

# %% Fase 1 — entrenar en train y evaluar en validación
rows = []
for nombre_dataset, d in datos.items():
    for name, (algoritmo, params, estimator) in candidates.items():
        model = clone(estimator)
        inicio = time.perf_counter()
        model.fit(d["X_train"], d["y_train"])
        tiempo = time.perf_counter() - inicio

        fila = metric_row(name, d["y_val"], model.predict(d["X_val"]))
        fila["Dataset"] = nombre_dataset
        fila["Algoritmo"] = algoritmo
        fila["F1 train"] = f1_score(d["y_train"], model.predict(d["X_train"]), average="macro", zero_division=0)
        fila["Tiempo fit (s)"] = tiempo
        fila.update(params)
        rows.append(fila)

resultados_fase1 = pd.DataFrame(rows)
print("Fase 1 lista:", len(resultados_fase1), "configuraciones evaluadas.")

# %% mejor configuración de cada modelo (validación)
columnas_tabla = ["Modelo", "Accuracy", "Precision", "Recall", "F1", "F1 train"]
mejores_por_algoritmo = {}

for nombre_dataset in datos:
    tabla = resultados_fase1[resultados_fase1["Dataset"] == nombre_dataset]
    # sort estable: si hay empate se conserva el primero de la tabla
    tabla = tabla.sort_values("F1", ascending=False, kind="stable")
    mejores = tabla.drop_duplicates("Algoritmo")       # primera fila de cada algoritmo = su mejor configuración
    mejores_por_algoritmo[nombre_dataset] = mejores
    print(f"\n{nombre_dataset}: mejor configuración de cada modelo")
    display(mejores[columnas_tabla].set_index("Modelo").round(3))

# %% Elegir los 2 mejores modelos de la Fase 1
modelos_cv = {}

for nombre_dataset, mejores in mejores_por_algoritmo.items():
    sin_base = mejores[mejores["Algoritmo"] != "Base"]
    top2 = sin_base.sort_values("F1", ascending=False, kind="stable").head(2)

    modelos_cv[nombre_dataset] = top2["Modelo"].tolist()

    print(f"\n{nombre_dataset}: dos mejores modelos de la Fase 1")
    display(top2[["Algoritmo", "Modelo", "F1", "F1 train"]].round(3))

###############################################################################
# 6. Fase 2: validación cruzada K-Fold con los dos mejores candidatos de cada dataset
###############################################################################

# %% validación cruzada K-Fold (distintas particiones)
def crear_cv_splits(d, k, semilla, con_grupos):
    if con_grupos:
        cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=semilla)
        return list(cv.split(d["X_train"], d["y_train"], groups=d["grupos_train"]))
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=semilla)
    return list(cv.split(d["X_train"], d["y_train"]))

# Diez repeticiones con semillas diferentes a la exploración inicial.
REPETICIONES_CV = 10
K_SOLICITADO = 5
semillas_cv = [SEED + 100 + i for i in range(REPETICIONES_CV)]

filas_cv = []
for nombre_dataset, d in datos.items():
    grupos_clase = pd.DataFrame({"gama": d["y_train"], "grupo": d["grupos_train"]}).drop_duplicates()
    minimo_grupos = int(grupos_clase.groupby("gama")["grupo"].nunique().min())
    k = min(K_SOLICITADO, minimo_grupos)
    if k < 2:
        print(f"{nombre_dataset}: no es posible hacer CV estratificada por grupos; alguna gama tiene menos de 2 grupos.")
        continue
    for name in modelos_cv[nombre_dataset]:
        for semilla in semillas_cv:
            cv_splits = crear_cv_splits(d, k, semilla, con_grupos=True)
            scores = cross_validate(clone(candidates[name][2]), d["X_train"], d["y_train"],
                                    cv=cv_splits, scoring=scoring)
            for fold in range(len(cv_splits)):
                filas_cv.append({"Dataset": nombre_dataset, "Modelo": name, "K": k, "Repeticion": semilla - (SEED + 100) + 1, "Semilla": semilla,
                                 "Fold": fold + 1,
                                 "Accuracy": scores["test_Accuracy"][fold],
                                 "Precision": scores["test_Precision"][fold],
                                 "Recall": scores["test_Recall"][fold],
                                 "F1": scores["test_F1"][fold]})

resultados_fase2 = pd.DataFrame(filas_cv)
print("Validación cruzada lista:", len(resultados_fase2), "folds evaluados.")

# %% resultados de la validación cruzada (media y desviación entre folds)
metricas = ["Accuracy", "Precision", "Recall", "F1"]
ganadores = {}

for nombre_dataset in datos:
    tabla_cv = resultados_fase2[resultados_fase2["Dataset"] == nombre_dataset]
    print(f"\n{nombre_dataset}: media (+/- desviación) de todos los folds")
    resumen = tabla_cv.groupby("Modelo", sort=False)[metricas].agg(["mean", "std"]).round(3)
    display(resumen)

    print("Media por tipo de partición (K y semilla)")
    display(tabla_cv.groupby(["Modelo", "K", "Semilla"], sort=False)[metricas].mean().round(3))

    # El ganador es el de mayor F1 medio; si hay empate se conserva el primero (el mejor de la Fase 1)
    f1_medio = tabla_cv.groupby("Modelo", sort=False)["F1"].mean()
    ganadores[nombre_dataset] = f1_medio.idxmax()
    print(f"Modelo elegido para {nombre_dataset}: {ganadores[nombre_dataset]}")

# --- 6.1 Resultados fold por fold y estabilidad de métricas ---

# %%
display(resultados_fase2.sort_values(["Dataset", "Modelo", "Repeticion", "Fold"]).head(60))
for nombre in datos:
    sub = resultados_fase2[resultados_fase2["Dataset"] == nombre]
    for metrica in ["Accuracy", "Precision", "Recall", "F1"]:
        plt.figure(figsize=(12, 4))
        sns.boxplot(data=sub, x="Modelo", y=metrica)
        plt.xticks(rotation=70, ha="right")
        plt.title(f"{nombre}: distribución de {metrica} en validación cruzada")
        plt.tight_layout(); plt.show()

# --- 6.2 Sensibilidad, falsos positivos y brecha train-validación ---

# %%
from sklearn.preprocessing import label_binarize
filas_detalle = []
for nombre, d in datos.items():
    for modelo_nombre in modelos_cv[nombre]:
        est = clone(candidates[modelo_nombre][2])
        inicio = time.perf_counter()
        est.fit(d["X_train"], d["y_train"])
        tiempo_fit = time.perf_counter() - inicio
        pred_train = est.predict(d["X_train"])
        pred_val = est.predict(d["X_val"])
        f1_train = f1_score(d["y_train"], pred_train, average="macro", zero_division=0)
        f1_val = f1_score(d["y_val"], pred_val, average="macro", zero_division=0)
        cm = confusion_matrix(d["y_val"], pred_val, labels=GAMAS)
        total_tn = total_fp = total_fn = total_tp = 0
        especificidades = []; falsas_alarmas = []
        for j, clase in enumerate(GAMAS):
            tp = int(cm[j, j]); fn = int(cm[j, :].sum() - tp)
            fp = int(cm[:, j].sum() - tp); tn = int(cm.sum() - tp - fn - fp)
            total_tn += tn; total_fp += fp; total_fn += fn; total_tp += tp
            especificidades.append(tn/(tn+fp) if (tn+fp) else np.nan)
            falsas_alarmas.append(100*fp/(fp+tn) if (fp+tn) else np.nan)
        filas_detalle.append({"Dataset": nombre, "Modelo": modelo_nombre,
            "TN acumulados OvR": total_tn, "FP acumulados OvR": total_fp,
            "FN acumulados OvR": total_fn, "TP acumulados OvR": total_tp,
            "Especificidad macro": np.nanmean(especificidades),
            "Falsas alarmas por 100 (macro)": np.nanmean(falsas_alarmas),
            "F1 train": f1_train, "F1 validación": f1_val,
            "Brecha F1 train-validación": f1_train-f1_val, "Tiempo fit (s)": tiempo_fit})
detalle_metricas = pd.DataFrame(filas_detalle)
display(detalle_metricas.round(3))
for nombre in datos:
    sub = detalle_metricas[detalle_metricas["Dataset"] == nombre]
    plt.figure(figsize=(9, 4))
    sns.boxplot(data=resultados_fase2[resultados_fase2["Dataset"] == nombre], x="Modelo", y="F1")
    plt.xticks(rotation=70, ha="right")
    plt.title(f"{nombre}: F1 por fold de validación cruzada")
    plt.tight_layout(); plt.show()

# %% Conclusiones de la Fase 2 (resultados numéricos automáticos)
from IPython.display import Markdown, display

lineas = ["## Conclusiones de la Fase 2: validación cruzada", ""]

if resultados_fase2.empty:
    lineas += ["No se generaron resultados de validación cruzada. Revisa que la celda anterior haya terminado correctamente."]
else:
    for nombre_dataset in datos:
        sub = resultados_fase2[resultados_fase2["Dataset"] == nombre_dataset].copy()
        if sub.empty:
            lineas += [f"### {nombre_dataset}", "No hay folds disponibles para este dataset.", ""]
            continue

        resumen = sub.groupby("Modelo")["F1"].agg(media="mean", desviacion="std").fillna(0).sort_values("media", ascending=False)
        ganador = resumen.index[0]
        media_ganador = float(resumen.loc[ganador, "media"])
        std_ganador = float(resumen.loc[ganador, "desviacion"])
        mas_estable = resumen["desviacion"].idxmin()
        media_estable = float(resumen.loc[mas_estable, "media"])
        std_estable = float(resumen.loc[mas_estable, "desviacion"])

        if len(resumen) > 1:
            segundo = resumen.index[1]
            media_segundo = float(resumen.loc[segundo, "media"])
            std_segundo = float(resumen.loc[segundo, "desviacion"])
            brecha = media_ganador - media_segundo
            variacion = max(std_ganador, std_segundo)
            if brecha < variacion:
                lectura = "La diferencia es menor que la desviación estándar más alta entre ambos modelos; por tanto, no constituye evidencia clara de superioridad."
            else:
                lectura = "La diferencia supera la desviación estándar más alta entre ambos modelos, aunque debe interpretarse como estabilidad descriptiva y no como una prueba estadística formal."
        else:
            segundo, media_segundo, std_segundo, brecha, lectura = "N/A", float("nan"), float("nan"), float("nan"), "Solo hay un modelo disponible para comparar."

        # Comprobar si la preferencia por el ganador se repite en las 10 semillas.
        por_repeticion = sub.groupby(["Semilla", "Modelo"])["F1"].mean().unstack("Modelo")
        victorias = int((por_repeticion.idxmax(axis=1) == ganador).sum()) if not por_repeticion.empty else 0
        total_repeticiones = int(por_repeticion.shape[0])
        inicial = modelos_cv.get(nombre_dataset, ["N/D"])[0]
        cambio = "se mantuvo" if ganador == inicial else f"cambió de {inicial} a {ganador}"

        lineas += [
            f"### {nombre_dataset}",
            f"- **Mejor F1 macro medio:** {ganador}, con **{media_ganador:.3f} ± {std_ganador:.3f}** (media ± desviación estándar entre folds).",
            f"- **Más estable:** {mas_estable}, con desviación estándar **{std_estable:.3f}** y F1 medio **{media_estable:.3f}**.",
            f"- **Comparación con el segundo:** {segundo} obtuvo F1 medio **{media_segundo:.3f} ± {std_segundo:.3f}**; la diferencia entre medias fue **{brecha:.3f}**. {lectura}",
            f"- El modelo con mayor media fue el mejor en **{victorias} de {total_repeticiones}** repeticiones por semilla. Frente al candidato que encabezó la Fase 1, la selección **{cambio}**.",
            ""
        ]

    lineas += ["**Conclusión general:** se elige por dataset el modelo con mayor F1 macro medio en validación cruzada, pero también se reporta la desviación estándar. Si dos medias son cercanas en relación con la variación entre folds, la diferencia no debe presentarse como concluyente. El test final se usa después como evaluación independiente."]

display(Markdown("\n".join(lineas)))

###############################################################################
# 7. Fase 3: gráficas
###############################################################################

# -----------------------------------------------------------------------------
# 7.1 Comparación de métricas
# -----------------------------------------------------------------------------

# %% comparación de métricas por modelo
for nombre_dataset in datos:
    mejores = mejores_por_algoritmo[nombre_dataset]
    comparacion = mejores.melt(id_vars="Algoritmo", value_vars=metricas, var_name="Métrica", value_name="Valor")

    plt.figure(figsize=(10, 4))
    sns.barplot(data=comparacion, x="Algoritmo", y="Valor", hue="Métrica")
    plt.title(f"Comparación de métricas en validación: {nombre_dataset}")
    plt.ylabel("Valor (promedio macro)")
    plt.ylim(0, 1)
    plt.show()

# -----------------------------------------------------------------------------
# 7.2 Matrices de confusión por modelo
# -----------------------------------------------------------------------------

# %% matrices de confusión
nombres_cortos = [g.replace("Gama ", "") for g in GAMAS]

for nombre_dataset, d in datos.items():
    mejores = mejores_por_algoritmo[nombre_dataset]
    mejores = mejores[mejores["Algoritmo"] != "Base"]

    plt.figure(figsize=(12, 10))
    for i, (_, fila) in enumerate(mejores.iterrows()):
        model = clone(candidates[fila["Modelo"]][2])
        model.fit(d["X_train"], d["y_train"])
        ax = plt.subplot(2, 2, i + 1)
        ConfusionMatrixDisplay.from_predictions(d["y_val"], model.predict(d["X_val"]), labels=GAMAS,
                                                display_labels=nombres_cortos, cmap="Blues",
                                                colorbar=False, ax=ax)
        ax.set_title(fila["Modelo"])
    plt.suptitle(f"Matrices de confusión en validación: {nombre_dataset}")
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------
# 7.3 Impacto de los hiperparámetros
# -----------------------------------------------------------------------------

# %% efecto de los hiperparámetros en el F1 de validación
for nombre_dataset in datos:
    t = resultados_fase1[resultados_fase1["Dataset"] == nombre_dataset]

    plt.figure(figsize=(14, 9))

    plt.subplot(2, 2, 1)
    sub = t[t["Algoritmo"] == "GNB"].copy()
    sns.pointplot(data=sub, x="var_smoothing", y="F1", errorbar=None)
    plt.title("GNB: var_smoothing")

    plt.subplot(2, 2, 2)
    sub = t[t["Algoritmo"] == "DT"].copy()
    if not sub.empty:
        sub["min_samples_leaf"] = pd.to_numeric(sub["min_samples_leaf"], errors="coerce").astype(int)
        sns.pointplot(data=sub, x="max_depth", y="F1", hue="min_samples_leaf",
                      order=["1", "2", "4", "8", "None"], errorbar=None)
    plt.title("DT: max_depth y min_samples_leaf")

    plt.subplot(2, 2, 3)
    sub = t[t["Algoritmo"] == "RF"].copy()
    if not sub.empty:
        sub["n_estimators"] = pd.to_numeric(sub["n_estimators"], errors="coerce").astype(int)
        sns.pointplot(data=sub, x="n_estimators", y="F1", hue="max_depth", errorbar=None)
    plt.title("RF: n_estimators y max_depth")

    plt.subplot(2, 2, 4)
    sub = t[t["Algoritmo"] == "SVM"].copy()
    sns.pointplot(data=sub, x="C", y="F1", hue="kernel", errorbar=None)
    plt.title("SVM: C y kernel")

    plt.suptitle(f"Impacto de los hiperparámetros: {nombre_dataset}")
    plt.tight_layout()
    plt.show()

# %% sobreajuste del árbol y costo del bosque
# F1 de train vs validación según max_depth (árbol) y tiempo según n_estimators (bosque)
for nombre_dataset in datos:
    t = resultados_fase1[resultados_fase1["Dataset"] == nombre_dataset]

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    arbol = t[(t["Algoritmo"] == "DT") & (t["min_samples_leaf"] == 1)]
    plt.plot(arbol["max_depth"], arbol["F1 train"], marker="o", label="Train")
    plt.plot(arbol["max_depth"], arbol["F1"], marker="x", label="Validación")
    plt.xlabel("max_depth")
    plt.ylabel("F1 macro")
    plt.title("DT (min_samples_leaf=1): train vs validación")
    plt.legend()

    plt.subplot(1, 2, 2)
    bosque = t[(t["Algoritmo"] == "RF") & (t["max_depth"] == "None")]
    plt.plot(bosque["n_estimators"], bosque["Tiempo fit (s)"], marker="s", color="darkorange")
    plt.xlabel("n_estimators")
    plt.ylabel("Tiempo de entrenamiento (s)")
    plt.title("RF: costo de entrenamiento")

    plt.suptitle(nombre_dataset)
    plt.tight_layout()
    plt.show()

# %% feature importance
#  importancia de las variables en el mejor árbol de decisión (no demuestra causalidad)
for nombre_dataset, d in datos.items():
    fila = mejores_por_algoritmo[nombre_dataset].query("Algoritmo == 'DT'").iloc[0]
    dt = clone(candidates[fila["Modelo"]][2])
    dt.fit(d["X_train"], d["y_train"])

    nombres_vars = dt[:-1].get_feature_names_out()
    importance = pd.Series(dt[-1].feature_importances_, index=nombres_vars)
    print(f"{nombre_dataset} - {fila['Modelo']}")
    display(importance.sort_values(ascending=False).head(8).rename("Importancia").to_frame())

# --- 7.5 Visualización de un árbol de decisión ---

# %%
from sklearn.tree import plot_tree
for nombre_dataset, d in datos.items():
    fila = mejores_por_algoritmo[nombre_dataset].query("Algoritmo == 'DT'").iloc[0]
    dt_pipe = clone(candidates[fila["Modelo"]][2])
    dt_pipe.fit(d["X_train"], d["y_train"])
    nombres_vars = dt_pipe[:-1].get_feature_names_out()
    plt.figure(figsize=(20, 9))
    plot_tree(dt_pipe[-1], feature_names=nombres_vars, class_names=GAMAS,
              filled=False, rounded=True, max_depth=3, fontsize=7)
    plt.title(f"Árbol de decisión (hasta 3 niveles) — {nombre_dataset}")
    plt.tight_layout(); plt.show()

###############################################################################
# 8. Evaluación final en test
###############################################################################

# %% Final test — ejecutar después de fijar la configuración
resultados_finales = []

for nombre_dataset, d in datos.items():
    selected_name = ganadores[nombre_dataset]
    final_model = clone(candidates[selected_name][2])
    final_model.fit(d["X_dev"], d["y_dev"])
    y_test_pred = final_model.predict(d["X_test"])

    fila = metric_row(selected_name, d["y_test"], y_test_pred)
    fila["Dataset"] = nombre_dataset
    resultados_finales.append(fila)

    print("=" * 70)
    print(f"{nombre_dataset}: modelo elegido = {selected_name} | registros de test: {len(d['y_test'])}")
    display(pd.DataFrame([fila])[["Modelo", "Accuracy", "Precision", "Recall", "F1"]].round(3))
    print(classification_report(d["y_test"], y_test_pred, labels=GAMAS, zero_division=0))

    ConfusionMatrixDisplay.from_predictions(d["y_test"], y_test_pred, labels=GAMAS,
                                            display_labels=nombres_cortos, cmap="Blues")
    plt.title(f"Test final: {nombre_dataset} - {selected_name}")
    plt.show()

    # Productos que el modelo clasificó en otra gama
    mal = d["y_test"].to_numpy() != y_test_pred
    identificadores = [c for c in ["match_key", "name", "benchmark_model"] if c in d["df_test"].columns]
    errores = d["df_test"].loc[mal, identificadores].copy()
    errores["Gama real"] = d["y_test"].to_numpy()[mal]
    errores["Gama predicha"] = y_test_pred[mal]
    print("Ejemplos mal clasificados:", len(errores))
    display(errores.head(10))

resultados_finales_df = pd.DataFrame(resultados_finales).set_index("Dataset")
display(resultados_finales_df[["Modelo", "Accuracy", "Precision", "Recall", "F1"]].round(3))

###############################################################################
# AVANCE 3 — Clasificación binaria y regresión
###############################################################################

###############################################################################
# 9. Parte A: Clasificación binaria (gama baja vs. gama alta)
###############################################################################

# -----------------------------------------------------------------------------
# 9.1 Cómo se pasa de 5 gamas a 2 clases
# -----------------------------------------------------------------------------

# %% 9.1 crear la variable binaria y conservar la misma partición
from sklearn.metrics import roc_auc_score, roc_curve, RocCurveDisplay
from sklearn.dummy import DummyClassifier

CLASES_BIN = ["Gama baja", "Gama alta"]
POSITIVA = "Gama alta"
MAPA_BIN = {"Gama baja": "Gama baja", "Gama media-baja": "Gama baja",
            "Gama media-alta": "Gama alta", "Gama alta": "Gama alta"}   # "Gama media" no aparece: se descarta

datos_bin = {}
filas_conteo = []
for nombre, d in datos.items():
    nuevo = {}
    for parte in ["train", "val", "test", "dev"]:
        mantener = d[f"y_{parte}"].isin(MAPA_BIN.keys())          # quita la gama media
        nuevo[f"X_{parte}"] = d[f"X_{parte}"][mantener]
        nuevo[f"y_{parte}"] = d[f"y_{parte}"][mantener].map(MAPA_BIN)
        if parte == "train":
            nuevo["grupos_train"] = d["grupos_train"][mantener]
        if parte == "test":
            nuevo["df_test"] = d["df_test"].loc[nuevo["y_test"].index]
        conteo = nuevo[f"y_{parte}"].value_counts().reindex(CLASES_BIN, fill_value=0)
        filas_conteo.append({"Dataset": nombre, "Conjunto": parte, "Gama baja": conteo["Gama baja"],
                             "Gama alta": conteo["Gama alta"], "Total": int(conteo.sum())})
    datos_bin[nombre] = nuevo

display(pd.DataFrame(filas_conteo).pivot(index="Dataset", columns="Conjunto",
        values=["Gama baja", "Gama alta", "Total"]).reindex(["GPU", "CPU", "COOLER"]))

# -----------------------------------------------------------------------------
# 9.2 Fase 1 (binaria): los mismos cuatro algoritmos con distintos hiperparámetros
# -----------------------------------------------------------------------------

# %% 9.2 Fase 1 binaria: entrenar en train y evaluar en validación
filas = []
for nombre_dataset, d in datos_bin.items():
    for name, (algoritmo, params, estimator) in candidates.items():
        model = clone(estimator)
        inicio = time.perf_counter()
        model.fit(d["X_train"], d["y_train"])
        tiempo = time.perf_counter() - inicio
        fila = metric_row(name, d["y_val"], model.predict(d["X_val"]))
        fila["Dataset"] = nombre_dataset
        fila["Algoritmo"] = algoritmo
        fila["F1 train"] = f1_score(d["y_train"], model.predict(d["X_train"]), average="macro", zero_division=0)
        fila["Tiempo fit (s)"] = tiempo
        fila.update(params)
        filas.append(fila)

resultados_bin_f1 = pd.DataFrame(filas)
print("Fase 1 binaria lista:", len(resultados_bin_f1), "configuraciones evaluadas.")

mejores_bin = {}
for nombre_dataset in datos_bin:
    tabla = resultados_bin_f1[resultados_bin_f1["Dataset"] == nombre_dataset]
    tabla = tabla.sort_values("F1", ascending=False, kind="stable")
    mejores_bin[nombre_dataset] = tabla.drop_duplicates("Algoritmo")   # mejor configuración de cada algoritmo
    print(f"\n{nombre_dataset}: mejor configuración de cada algoritmo (validación, binaria)")
    display(mejores_bin[nombre_dataset][["Modelo", "Accuracy", "Precision", "Recall", "F1", "F1 train"]].set_index("Modelo").round(3))

# -----------------------------------------------------------------------------
# 9.3 Fase 2 (binaria): validación cruzada K-Fold
# -----------------------------------------------------------------------------

# %% 9.3 Fase 2 binaria: validación cruzada y elección de los 2 mejores
candidatos_bin = {nombre: mejores_bin[nombre][mejores_bin[nombre]["Algoritmo"] != "Base"]["Modelo"].tolist()
                  for nombre in datos_bin}

filas_cv = []
for nombre_dataset, d in datos_bin.items():
    grupos_clase = pd.DataFrame({"y": d["y_train"], "grupo": d["grupos_train"]}).drop_duplicates()
    k = min(K_SOLICITADO, int(grupos_clase.groupby("y")["grupo"].nunique().min()))
    for name in candidatos_bin[nombre_dataset]:
        for semilla in semillas_cv:
            cv_splits = crear_cv_splits(d, k, semilla, con_grupos=True)
            scores = cross_validate(clone(candidates[name][2]), d["X_train"], d["y_train"],
                                    cv=cv_splits, scoring=scoring)
            for fold in range(len(cv_splits)):
                filas_cv.append({"Dataset": nombre_dataset, "Modelo": name, "Algoritmo": candidates[name][0], "K": k,
                                 "Semilla": semilla, "Fold": fold + 1,
                                 "Accuracy": scores["test_Accuracy"][fold], "Precision": scores["test_Precision"][fold],
                                 "Recall": scores["test_Recall"][fold], "F1": scores["test_F1"][fold]})
resultados_bin_cv = pd.DataFrame(filas_cv)

mejores2_clasificacion = {}
for nombre_dataset in datos_bin:
    sub = resultados_bin_cv[resultados_bin_cv["Dataset"] == nombre_dataset]
    resumen = sub.groupby("Modelo")[["Accuracy", "Precision", "Recall", "F1"]].agg(["mean", "std"])
    resumen = resumen.sort_values([("F1", "mean"), ("F1", "std")], ascending=[False, True], kind="stable")   # empate: el más estable
    mejores2_clasificacion[nombre_dataset] = resumen.index[:2].tolist()
    print(f"\n{nombre_dataset}: validación cruzada (media y desviación entre folds)")
    display(resumen.round(3))
    print("Los 2 mejores modelos de clasificación binaria:", mejores2_clasificacion[nombre_dataset])

# %% 9.3 gráficas de la Fase 2 binaria: F1 por fold y comparación en validación
fig, ejes = plt.subplots(1, 3, figsize=(18, 4.5))
for ax, nombre_dataset in zip(ejes, datos_bin):
    sub = resultados_bin_cv[resultados_bin_cv["Dataset"] == nombre_dataset].copy()
    sub["Etiqueta"] = sub["Algoritmo"]
    orden = sub.groupby("Etiqueta")["F1"].mean().sort_values(ascending=False).index
    sns.boxplot(data=sub, x="Etiqueta", y="F1", order=orden, ax=ax)
    ax.set_title(f"{nombre_dataset}: F1 por fold (validación cruzada)")
    ax.set_xlabel(""); ax.set_ylim(0, 1.05)
plt.tight_layout(); plt.show()

for nombre_dataset in datos_bin:
    comparacion = mejores_bin[nombre_dataset].melt(id_vars="Algoritmo", value_vars=metricas,
                                                   var_name="Métrica", value_name="Valor")
    plt.figure(figsize=(10, 4))
    sns.barplot(data=comparacion, x="Algoritmo", y="Valor", hue="Métrica")
    plt.title(f"Clasificación binaria en validación: {nombre_dataset}")
    plt.ylim(0, 1.05); plt.ylabel("Valor (promedio macro)")
    plt.show()

# -----------------------------------------------------------------------------
# 9.4 Probar el modelo: evaluación final en test (binaria)
# -----------------------------------------------------------------------------

# %% 9.4 test final de los 2 mejores modelos de clasificación binaria
def puntaje_clase_alta(modelo, X):
    # Probabilidad (o puntaje de decisión) de pertenecer a la gama alta, para la curva ROC
    posicion = list(modelo.classes_).index(POSITIVA)
    if hasattr(modelo, "predict_proba"):
        return modelo.predict_proba(X)[:, posicion]
    puntaje = modelo.decision_function(X)           # SVM: positivo = classes_[1]
    return puntaje if posicion == 1 else -puntaje   # se ajusta el signo para que mayor = más gama alta

modelos_test_bin = {}       # modelo reentrenado, para usarlo después con ejemplos
filas_test_bin = []
for nombre_dataset, d in datos_bin.items():
    nombres = mejores2_clasificacion[nombre_dataset] + ["Base (gama más común)"]
    fig, ejes = plt.subplots(1, 3, figsize=(17, 4.5))
    for i, name in enumerate(nombres):
        modelo = clone(candidates[name][2]).fit(d["X_dev"], d["y_dev"])
        pred = modelo.predict(d["X_test"])
        puntaje = puntaje_clase_alta(modelo, d["X_test"])
        tn, fp, fn, tp = confusion_matrix(d["y_test"], pred, labels=CLASES_BIN).ravel()
        fila = metric_row(name, d["y_test"], pred)
        fila.update({"Dataset": nombre_dataset, "Puesto": (i + 1) if i < 2 else "Base",
                     "Especificidad": tn / (tn + fp) if (tn + fp) else np.nan,
                     "AUC": roc_auc_score(d["y_test"] == POSITIVA, puntaje) if i < 2 else 0.5,
                     "Aciertos": int(tn + tp), "Errores": int(fp + fn), "Test (n)": len(d["y_test"])})
        filas_test_bin.append(fila)
        if i < 2:
            modelos_test_bin[(nombre_dataset, i)] = modelo
            ConfusionMatrixDisplay.from_predictions(d["y_test"], pred, labels=CLASES_BIN, cmap="Blues",
                                                    colorbar=False, ax=ejes[i])
            ejes[i].set_title(f"{nombre_dataset} — puesto {i + 1}\n{name}", fontsize=9)
            RocCurveDisplay.from_predictions(d["y_test"] == POSITIVA, puntaje, name=f"Puesto {i + 1}: {candidates[name][0]}", ax=ejes[2])
    ejes[2].plot([0, 1], [0, 1], "k--", label="Azar")
    ejes[2].set_title(f"Curva ROC en test: {nombre_dataset}"); ejes[2].legend(fontsize=8)
    plt.tight_layout(); plt.show()

resultados_test_bin = pd.DataFrame(filas_test_bin)
columnas_t = ["Dataset", "Puesto", "Modelo", "Accuracy", "Precision", "Recall", "Especificidad", "F1", "AUC", "Aciertos", "Errores", "Test (n)"]
display(resultados_test_bin[columnas_t].round(3).set_index(["Dataset", "Puesto"]))

# %% 9.4 probar el modelo con productos de test (real vs. predicho)
# Se muestran 12 productos de test del mejor modelo de cada dataset: la gama real, la que predijo
# el modelo y qué tan seguro estuvo (probabilidad de gama alta).
for nombre_dataset, d in datos_bin.items():
    modelo = modelos_test_bin[(nombre_dataset, 0)]
    pred = modelo.predict(d["X_test"])
    prob = puntaje_clase_alta(modelo, d["X_test"])
    ejemplos = d["df_test"][[c for c in ["name", "match_key"] if c in d["df_test"].columns][:1]].copy()
    ejemplos["Gama real"] = d["y_test"].to_numpy()
    ejemplos["Gama predicha"] = pred
    ejemplos["Prob. gama alta" if hasattr(modelo, "predict_proba") else "Puntaje gama alta (SVM)"] = np.round(prob, 2)
    ejemplos["¿Acertó?"] = np.where(ejemplos["Gama real"] == ejemplos["Gama predicha"], "sí", "NO")
    print(f"\n{nombre_dataset}: mejor modelo = {mejores2_clasificacion[nombre_dataset][0]}")
    display(ejemplos.sample(min(12, len(ejemplos)), random_state=SEED).reset_index(drop=True))
    print("Productos mal clasificados en test:", int((pred != d["y_test"].to_numpy()).sum()), "de", len(pred))

# -----------------------------------------------------------------------------
# 9.5 Lectura de los resultados: ¿por qué la clasificación binaria sale tan alta?
# -----------------------------------------------------------------------------

# %% 9.5 regla de un solo corte y qué hace el modelo con la gama media
from sklearn.tree import export_text

print("(1) Árbol de un solo nivel (entrenado con train + validación, probado en test)\n")
for nombre_dataset, d in datos_bin.items():
    corte = make_pipeline(SimpleImputer(strategy="median"), DecisionTreeClassifier(max_depth=1, random_state=SEED))
    corte.fit(d["X_dev"], d["y_dev"])
    acc = accuracy_score(d["y_test"], corte.predict(d["X_test"]))
    f1m = f1_score(d["y_test"], corte.predict(d["X_test"]), average="macro", zero_division=0)
    print(f"{nombre_dataset}: Accuracy en test = {acc:.3f} | F1 macro = {f1m:.3f}")
    print(export_text(corte[-1], feature_names=list(corte[:-1].get_feature_names_out())))

print("(2) Productos de GAMA MEDIA de test (no se usaron para entrenar): ¿hacia dónde los manda el mejor modelo?")
filas = []
for nombre_dataset, d in datos.items():
    mascara = d["y_test"] == "Gama media"
    modelo = modelos_test_bin[(nombre_dataset, 0)]
    pred = pd.Series(modelo.predict(d["X_test"][mascara]))
    filas.append({"Dataset": nombre_dataset, "Mejor modelo": mejores2_clasificacion[nombre_dataset][0],
                  "Productos de gama media en test": int(mascara.sum()),
                  "% clasificados como gama alta": round(100 * (pred == POSITIVA).mean(), 1),
                  "% clasificados como gama baja": round(100 * (pred == "Gama baja").mean(), 1)})
display(pd.DataFrame(filas).set_index("Dataset"))

###############################################################################
# 10. Parte B: Regresión (predecir el puntaje del benchmark)
###############################################################################

# -----------------------------------------------------------------------------
# 10.1 Variable objetivo continua
# -----------------------------------------------------------------------------

# %% 10.1 variable objetivo continua y partición (la misma del Avance 2)
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

OBJETIVO_REG = {"GPU": "benchmark_score", "CPU": "benchmark_score", "COOLER": "score_percentile"}

datos_reg = {}
for nombre, d in datos.items():
    df0 = datasets_clasificacion[nombre].reset_index(drop=True)
    y_todo = pd.to_numeric(df0[OBJETIVO_REG[nombre]], errors="coerce")
    r = {k: d[k] for k in ["X_train", "X_val", "X_test", "X_dev", "grupos_train", "df_test"]}
    for parte in ["train", "val", "test", "dev"]:
        r[f"y_{parte}"] = y_todo.loc[d[f"X_{parte}"].index]
    r["gama_train"] = d["y_train"]            # solo para estratificar los folds de la validación cruzada
    datos_reg[nombre] = r
    print(f"{nombre}: y = {OBJETIVO_REG[nombre]} | train {len(r['y_train'])} · val {len(r['y_val'])} · test {len(r['y_test'])}"
          f" | min {y_todo.min():.2f} · mediana {y_todo.median():.2f} · max {y_todo.max():.2f} | vacíos en y: {int(y_todo.isna().sum())}")

fig, ejes = plt.subplots(1, 3, figsize=(16, 3.8))
for ax, (nombre, r) in zip(ejes, datos_reg.items()):
    sns.histplot(r["y_dev"], kde=True, ax=ax)
    ax.set_title(f"{nombre}: distribución de {OBJETIVO_REG[nombre]}")
plt.tight_layout(); plt.show()

# -----------------------------------------------------------------------------
# 10.2 Fase 1 (regresión): cuatro algoritmos con distintos hiperparámetros
# -----------------------------------------------------------------------------

# %% 10.2 crear los regresores con sus hiperparámetros
def envolver(modelo, escalar=False):
    # Imputa con la mediana (aprendida solo en train), escala si hace falta y estandariza y
    pasos = [SimpleImputer(strategy="median")]
    if escalar:
        pasos.append(StandardScaler())
    pasos.append(modelo)
    return TransformedTargetRegressor(regressor=make_pipeline(*pasos), transformer=StandardScaler())

def crear_regresores():
    c = {}
    c["Base (promedio)"] = ("Base", {}, envolver(DummyRegressor(strategy="mean")))
    c["Lineal OLS"] = ("Lineal", {"alpha": "0"}, envolver(LinearRegression(), escalar=True))
    for alpha in [1, 10, 100]:
        c[f"Lineal Ridge alpha={alpha}"] = ("Lineal", {"alpha": str(alpha)}, envolver(Ridge(alpha=alpha), escalar=True))
    for depth in [2, 4, 6, 8, None]:
        for leaf in [1, 2, 5]:
            c[f"DT depth={depth} leaf={leaf}"] = ("DT", {"max_depth": str(depth), "min_samples_leaf": leaf},
                envolver(DecisionTreeRegressor(max_depth=depth, min_samples_leaf=leaf, random_state=SEED)))
    for n in [100, 200]:
        for depth in [8, None]:
            for mf in [1.0, "sqrt", 0.5]:
                c[f"RF n={n} depth={depth} features={mf}"] = ("RF",
                    {"n_estimators": n, "max_depth": str(depth), "max_features": str(mf)},
                    envolver(RandomForestRegressor(n_estimators=n, max_depth=depth, max_features=mf,
                                                   random_state=SEED, n_jobs=-1)))
    for C in [0.1, 1, 10, 100]:
        for eps in [0.01, 0.1]:
            c[f"SVR rbf C={C} epsilon={eps}"] = ("SVR", {"C": C, "epsilon": eps},
                envolver(SVR(kernel="rbf", C=C, epsilon=eps), escalar=True))
    return c

regresores = crear_regresores()
print("Configuraciones de regresión por dataset:", len(regresores))

def metricas_reg(nombre, y_true, y_pred):
    return {"Modelo": nombre, "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)), "R2": r2_score(y_true, y_pred)}

# %% 10.2 Fase 1 regresión: entrenar en train y evaluar en validación
filas = []
for nombre_dataset, r in datos_reg.items():
    for name, (algoritmo, params, estimator) in regresores.items():
        model = clone(estimator)
        inicio = time.perf_counter()
        model.fit(r["X_train"], r["y_train"])
        tiempo = time.perf_counter() - inicio
        fila = metricas_reg(name, r["y_val"], model.predict(r["X_val"]))
        fila.update({"Dataset": nombre_dataset, "Algoritmo": algoritmo,
                     "R2 train": r2_score(r["y_train"], model.predict(r["X_train"])), "Tiempo fit (s)": tiempo})
        fila.update(params)
        filas.append(fila)
resultados_reg_f1 = pd.DataFrame(filas)
print("Fase 1 de regresión lista:", len(resultados_reg_f1), "configuraciones evaluadas.")

mejores_reg = {}
for nombre_dataset in datos_reg:
    tabla = resultados_reg_f1[resultados_reg_f1["Dataset"] == nombre_dataset]
    tabla = tabla.sort_values(["R2", "MAE"], ascending=[False, True], kind="stable")
    mejores_reg[nombre_dataset] = tabla.drop_duplicates("Algoritmo")
    print(f"\n{nombre_dataset}: mejor configuración de cada algoritmo (validación)")
    display(mejores_reg[nombre_dataset][["Modelo", "MAE", "RMSE", "R2", "R2 train"]].set_index("Modelo").round(3))

# -----------------------------------------------------------------------------
# 10.3 Fase 2 (regresión): validación cruzada y elección de los 2 mejores
# -----------------------------------------------------------------------------

# %% 10.3 Fase 2 regresión: validación cruzada y elección de los 2 mejores
scoring_reg = {"R2": "r2", "MAE": "neg_mean_absolute_error", "RMSE": "neg_root_mean_squared_error"}
candidatos_reg = {n: mejores_reg[n][mejores_reg[n]["Algoritmo"] != "Base"]["Modelo"].tolist() for n in datos_reg}

filas_cv = []
for nombre_dataset, r in datos_reg.items():
    grupos_clase = pd.DataFrame({"g": r["gama_train"], "grupo": r["grupos_train"]}).drop_duplicates()
    k = min(K_SOLICITADO, int(grupos_clase.groupby("g")["grupo"].nunique().min()))
    for name in candidatos_reg[nombre_dataset]:
        for semilla in semillas_cv:
            cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=semilla)
            splits = list(cv.split(r["X_train"], r["gama_train"], groups=r["grupos_train"]))
            sc = cross_validate(clone(regresores[name][2]), r["X_train"], r["y_train"], cv=splits, scoring=scoring_reg)
            for fold in range(len(splits)):
                filas_cv.append({"Dataset": nombre_dataset, "Modelo": name, "Algoritmo": regresores[name][0], "K": k,
                                 "Semilla": semilla, "Fold": fold + 1, "R2": sc["test_R2"][fold],
                                 "MAE": -sc["test_MAE"][fold], "RMSE": -sc["test_RMSE"][fold]})
resultados_reg_cv = pd.DataFrame(filas_cv)

mejores2_regresion = {}
for nombre_dataset in datos_reg:
    sub = resultados_reg_cv[resultados_reg_cv["Dataset"] == nombre_dataset]
    resumen = sub.groupby("Modelo")[["R2", "MAE", "RMSE"]].agg(["mean", "std"]).sort_values([("R2", "mean"), ("R2", "std")], ascending=[False, True], kind="stable")
    mejores2_regresion[nombre_dataset] = resumen.index[:2].tolist()
    print(f"\n{nombre_dataset}: validación cruzada (media y desviación entre folds)")
    display(resumen.round(3))
    print("Los 2 mejores modelos de regresión:", mejores2_regresion[nombre_dataset])

# %% 10.3 gráficas de la regresión: R² por fold, validación y sobreajuste
fig, ejes = plt.subplots(1, 3, figsize=(18, 4.5))
for ax, nombre_dataset in zip(ejes, datos_reg):
    sub = resultados_reg_cv[resultados_reg_cv["Dataset"] == nombre_dataset]
    orden = sub.groupby("Algoritmo")["R2"].mean().sort_values(ascending=False).index
    sns.boxplot(data=sub, x="Algoritmo", y="R2", order=orden, ax=ax)
    ax.set_title(f"{nombre_dataset}: R² por fold (validación cruzada)"); ax.set_xlabel("")
plt.tight_layout(); plt.show()

fig, ejes = plt.subplots(1, 3, figsize=(18, 4))
for ax, nombre_dataset in zip(ejes, datos_reg):
    t = mejores_reg[nombre_dataset].copy()
    t = t.melt(id_vars="Algoritmo", value_vars=["R2 train", "R2"], var_name="Conjunto", value_name="R²")
    t["Conjunto"] = t["Conjunto"].replace({"R2 train": "Train", "R2": "Validación"})
    sns.barplot(data=t, x="Algoritmo", y="R²", hue="Conjunto", ax=ax)
    ax.set_title(f"{nombre_dataset}: R² train vs validación"); ax.set_xlabel("")
plt.tight_layout(); plt.show()

# -----------------------------------------------------------------------------
# 10.4 Probar los 2 mejores regresores: evaluación final en test
# -----------------------------------------------------------------------------

# %% 10.4 test final de los 2 mejores modelos de regresión
filas_test_reg = []
modelos_test_reg = {}
for nombre_dataset, r in datos_reg.items():
    nombres = mejores2_regresion[nombre_dataset] + ["Base (promedio)"]
    fig, ejes = plt.subplots(2, 2, figsize=(12, 9))
    for i, name in enumerate(nombres):
        modelo = clone(regresores[name][2]).fit(r["X_dev"], r["y_dev"])
        pred = modelo.predict(r["X_test"])
        fila = metricas_reg(name, r["y_test"], pred)
        fila.update({"Dataset": nombre_dataset, "Puesto": (i + 1) if i < 2 else "Base", "Test (n)": len(r["y_test"])})
        filas_test_reg.append(fila)
        if i < 2:
            modelos_test_reg[(nombre_dataset, i)] = modelo
            ejes[0, i].scatter(r["y_test"], pred, alpha=0.5)
            lim = [min(r["y_test"].min(), pred.min()), max(r["y_test"].max(), pred.max())]
            ejes[0, i].plot(lim, lim, "r--", label="Predicción perfecta")
            ejes[0, i].set_title(f"Puesto {i + 1}: {name}\nR²={fila['R2']:.3f} · MAE={fila['MAE']:.3g}", fontsize=9)
            ejes[0, i].set_xlabel("Real"); ejes[0, i].set_ylabel("Predicho"); ejes[0, i].legend()
            sns.histplot(r["y_test"].to_numpy() - pred, kde=True, ax=ejes[1, i])
            ejes[1, i].axvline(0, color="red", linestyle="--")
            ejes[1, i].set_title("Residuos (real − predicho)"); ejes[1, i].set_xlabel("Error")
    plt.suptitle(f"Test final de regresión: {nombre_dataset} (y = {OBJETIVO_REG[nombre_dataset]})")
    plt.tight_layout(); plt.show()

resultados_test_reg = pd.DataFrame(filas_test_reg)
display(resultados_test_reg[["Dataset", "Puesto", "Modelo", "MAE", "RMSE", "R2", "Test (n)"]].round(3).set_index(["Dataset", "Puesto"]))

# %% 10.4 variables más importantes del mejor Random Forest de regresión
for nombre_dataset, r in datos_reg.items():
    nombre_rf = mejores_reg[nombre_dataset].query("Algoritmo == 'RF'")["Modelo"].iloc[0]
    rf = clone(regresores[nombre_rf][2]).fit(r["X_dev"], r["y_dev"])
    pipe = rf.regressor_
    importancia = pd.Series(pipe[-1].feature_importances_, index=pipe[:-1].get_feature_names_out()).sort_values(ascending=False)
    print(f"{nombre_dataset} — {nombre_rf}")
    display(importancia.head(6).rename("Importancia").round(3).to_frame())

###############################################################################
# 11. Resumen del Avance 3: los 2 mejores modelos de clasificación y de regresión
###############################################################################

# %% 11. tabla final: 2 mejores modelos de clasificación y 2 de regresión por dataset
filas = []
for nombre_dataset in datos:
    for i in range(2):
        c = resultados_test_bin[(resultados_test_bin["Dataset"] == nombre_dataset) & (resultados_test_bin["Puesto"] == i + 1)].iloc[0]
        g = resultados_test_reg[(resultados_test_reg["Dataset"] == nombre_dataset) & (resultados_test_reg["Puesto"] == i + 1)].iloc[0]
        filas.append({"Dataset": nombre_dataset, "Puesto": i + 1,
                      "Clasificación binaria": c["Modelo"], "F1 test": round(c["F1"], 3), "Accuracy test": round(c["Accuracy"], 3),
                      "Regresión": g["Modelo"], "R² test": round(g["R2"], 3), "MAE test": round(g["MAE"], 3)})
resumen_avance3 = pd.DataFrame(filas).set_index(["Dataset", "Puesto"])
display(resumen_avance3)
