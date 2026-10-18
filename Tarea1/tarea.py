"""

    *****************************************************

ETAPA PREVIA - CARGA Y PARTICION DE DATOS
Proyecto: Regresion Logistica - Clasificacion de Diabetes
Dataset : Pima Indians Diabetes (Kaggle, 768 observaciones)
---------------------------------------------------------------------
ESTRUCTURA DEL PROYECTO
    Tarea1/
      |-- data/       datos crudos y particiones generadas
      |-- outputs/    figuras y tablas de todas las actividades
      |-- tarea.py    este script
      `-- 01_...py    scripts de las actividades siguientes
---------------------------------------------------------------------
Particion : 200 entrenamiento / 332 prueba / 236 no usados
---------------------------------------------------------------------
DICCIONARIO DE VARIABLES
    npreg    numero de embarazos previos              (conteo)
    glu      glucosa plasmatica tras prueba OGTT      (mg/dL)
    bp       presion arterial diastolica              (mm Hg)
    skin     pliegue cutaneo tricipital               (mm)
    insulin  insulina serica a 2 horas                (micro U/mL)
    bmi      indice de masa corporal                  (kg/m^2)
    ped      funcion pedigree de diabetes             (adimensional)
    age      edad de la paciente                      (años)
    type     diagnostico segun criterios OMS          (0 = no, 1 = si)

    *****************************************************

"""

# ---------------------------------------------------------------------
# 1. IMPORTACIONES
# ---------------------------------------------------------------------
from pathlib import Path                  # rutas portables entre sistemas
import pandas as pd                       # tablas de datos
import numpy as np                        # operaciones numericas y NaN
import matplotlib.pyplot as plt           # graficas base
import seaborn as sns                     # graficas estadisticas
from sklearn.model_selection import train_test_split   # particion de datos

# ---------------------------------------------------------------------
# 2. CONFIGURACION Y RUTAS RELATIVAS
# ---------------------------------------------------------------------
# __file__ es la ruta de ESTE script; .parent sube UN nivel.
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"              # carpeta de datos
OUT_DIR  = BASE_DIR / "outputs"           # carpeta de salidas
DATA_DIR.mkdir(exist_ok=True)             # crearlas si no existen
OUT_DIR.mkdir(exist_ok=True)

NOMBRE_CSV = "pima-indians-diabetes.csv"  # archivo original de Kaggle

# Ubicaciones donde se buscara el CSV, en orden de prioridad
RUTAS_CANDIDATAS = [
    DATA_DIR / NOMBRE_CSV,        # Tarea1/data/            <- recomendada
    BASE_DIR / NOMBRE_CSV,        # Tarea1/                 <- si esta suelto
    Path.cwd() / NOMBRE_CSV,      # carpeta de ejecucion
]

COLUMNAS = ["npreg", "glu", "bp", "skin", "insulin", "bmi", "ped", "age", "type"]

# Columnas donde un cero es fisiologicamente imposible (es un faltante disfrazado)
CERO_IMPOSIBLE = ["glu", "bp", "skin", "insulin", "bmi"]

TRAIN_SIZE  = 200                         # tamano de entrenamiento
TEST_SIZE   = 332                         # tamano de prueba
RANDOM_SEED = 42                          # semilla para reproducibilidad

sns.set_theme(style="whitegrid")          # estilo visual uniforme
pd.set_option("display.max_columns", None)# mostrar todas las columnas

# ---------------------------------------------------------------------
# 3. FUNCIONES
# ---------------------------------------------------------------------

def localizar_csv():
    """Busca el CSV en las rutas candidatas y devuelve la primera valida."""
    for ruta in RUTAS_CANDIDATAS:
        if ruta.exists():
            print(f"[RUTA] Archivo encontrado en: {ruta}")
            return ruta
    # ninguna ruta existe: detener con mensaje claro
    print("[ERROR] No se encontro el archivo en ninguna de estas rutas:")
    for ruta in RUTAS_CANDIDATAS:
        print(f"   - {ruta}")
    print(f"\nColoca '{NOMBRE_CSV}' en: {DATA_DIR}")
    raise FileNotFoundError(NOMBRE_CSV)


def cargar_datos():
    """Carga el CSV sin encabezados y asigna nombres de columna."""
    ruta = localizar_csv()
    # header=None: el archivo original no trae nombres en la primera fila
    df = pd.read_csv(ruta, header=None, names=COLUMNAS)
    print(f"[DATOS] Filas: {len(df)} | Columnas: {len(df.columns)}")
    return df


def corregir_ceros_imposibles(df):
    """
    Convierte ceros fisiologicamente imposibles en valores faltantes (NaN).

    Justificacion biologica:
    - Glucosa = 0  -> incompatible con la vida
    - Presion = 0  -> incompatible con la vida
    - IMC = 0      -> implicaria peso corporal cero
    - Piel = 0     -> pliegue cutaneo inexistente
    - Insulina = 0 -> sin medicion registrada

    Estos ceros no son datos reales: son ausencias codificadas como 0.
    """
    conteos = {}
    for col in CERO_IMPOSIBLE:
        conteos[col] = int((df[col] == 0).sum())      # contar ceros
        df.loc[df[col] == 0, col] = np.nan            # marcarlos como faltantes

    print("\n[CEROS CORREGIDOS A VALORES FALTANTES]")
    for col, n in conteos.items():
        print(f"  {col:<10}: {n}")
    print(f"  {'TOTAL':<10}: {sum(conteos.values())} celdas marcadas como faltantes")
    return df


def crear_particion(df):
    """
    Divide los datos en 200 entrenamiento y 332 prueba.

    Estratificacion: se conserva la misma proporcion de clases
    (diabetes / no diabetes) en ambos conjuntos. Sin esto, un
    conjunto podria quedar desbalanceado y sesgar las metricas.
    """
    X = df.drop(columns=["type"])     # predictores
    y = df["type"]                    # variable respuesta

    # Paso 1: extraer 200 filas para entrenamiento
    X_tr, X_resto, y_tr, y_resto = train_test_split(
        X, y,
        train_size=TRAIN_SIZE,
        stratify=y,
        random_state=RANDOM_SEED,
    )

    # Paso 2: del remanente, extraer 332 filas para prueba
    X_te, _, y_te, _ = train_test_split(
        X_resto, y_resto,
        train_size=TEST_SIZE,
        stratify=y_resto,
        random_state=RANDOM_SEED,
    )

    # Reconstruir tablas completas y reiniciar indices
    train = X_tr.assign(type=y_tr.values).reset_index(drop=True)
    test  = X_te.assign(type=y_te.values).reset_index(drop=True)
    return train, test


def guardar_salidas(train, test):
    """Guarda los conjuntos para que las etapas siguientes usen los mismos datos."""
    train.to_csv(DATA_DIR / "train.csv", index=False)
    test.to_csv(DATA_DIR / "test.csv", index=False)
    print(f"\n[GUARDADO] data/train.csv ({len(train)} filas)")
    print(f"[GUARDADO] data/test.csv  ({len(test)} filas)")


def resumen_clases(nombre, serie):
    """Imprime la distribucion de clases de forma explicita."""
    conteos = serie.value_counts().sort_index()   # indice=clase, valor=conteo
    total = len(serie)
    print(f"{nombre} (n = {total}):")
    for clase, cantidad in conteos.items():       # .items() da (indice, valor)
        porcentaje = 100 * cantidad / total
        print(f"  Clase {int(clase)}: {cantidad} casos ({porcentaje:.1f}%)")


def graficar_balance(train, test):
    """Grafica la cantidad de casos por clase en ambos conjuntos."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    sns.countplot(data=train, x="type", ax=ax1)
    ax1.set_title("Entrenamiento (200)")
    ax1.set_xlabel("Clase (0 = sin diabetes, 1 = diabetes)")

    sns.countplot(data=test, x="type", ax=ax2)
    ax2.set_title("Prueba (332)")
    ax2.set_xlabel("Clase (0 = sin diabetes, 1 = diabetes)")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "01_balance_clases.png", dpi=150)
    plt.show()


# ---------------------------------------------------------------------
# 4. EJECUCION PRINCIPAL
# ---------------------------------------------------------------------

if __name__ == "__main__":
    print("*" * 60)
    print(" ETAPA 1 - CARGA Y PARTICION DE DATOS")
    print("*" * 60)

    df = cargar_datos()                       # leer el CSV
    df = corregir_ceros_imposibles(df)        # tratar ceros invalidos
    train, test = crear_particion(df)         # dividir 200 / 332
    guardar_salidas(train, test)              # persistir los splits
    timeout = 10 #10 segundos

    print("\n" + "-" * 60)
    print(" RESUMEN ESTADISTICO - ENTRENAMIENTO")
    print("-" * 60)
    print(train.describe().T.round(2))
    timeout = 10 #10 segundos

    print("\n" + "-" * 60)
    print(" DISTRIBUCION DE CLASES")
    print("-" * 60)
    resumen_clases("Entrenamiento", train["type"])
    resumen_clases("Prueba", test["type"])
    timeout = 10 #10 segundos

    print("\n" + "-" * 60)
    print(" VALORES FALTANTES RESULTANTES")
    print("-" * 60)
    print(train.isna().sum())

    graficar_balance(train, test)

    print("\n" + "*" * 60)
    print(" ETAPA PREVIA COMPLETADA")
    print("*" * 60)
    print("PROXIMO PASO: Etapa 1 - graficas exploratorias")