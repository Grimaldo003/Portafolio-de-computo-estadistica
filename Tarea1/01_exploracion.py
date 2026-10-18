"""
=====================================================================
ETAPA 1 - ANALISIS EXPLORATORIO DE DATOS (EDA)
Proyecto: Regresion Logistica - Clasificacion de Diabetes
Requisito previo: haber ejecutado la Etapa Previa (genera train.csv)
=====================================================================
DICCIONARIO DE VARIABLES (dataset Pima Indians Diabetes)
---------------------------------------------------------------------
Variable  | Significado                                  | Unidades
----------|----------------------------------------------|----------
npreg     | Numero de embarazos previos                  | conteo
glu       | Glucosa plasmtica tras prueba OGTT           | mg/dL
          | (tolerancia oral a la glucosa, 2 horas)      |
bp        | Presion arterial diastolica                  | mm Hg
skin      | Grosor del pliegue cutaneo tricipital        | mm
          | (grasa subcutanea del brazo)                 |
insulin   | Insulina serica a 2 horas                    | micro U/mL
bmi       | Indice de masa corporal                      | kg/m^2
          | IMC = peso(kg) / altura(m)^2                |
ped       | Funcion pedigree de diabetes                 | adimensional
          | (proxy de antecedentes familiares)           |
age       | Edad de la paciente                          | anos
type      | Diagnostico segun criterios OMS             | 0 = no
          |                                              | 1 = si
----------|----------------------------------------------|----------
Nota: los ceros imposibles (glu, bp, skin, insulin, bmi)
fueron convertidos a NaN en la Etapa previa. Cada figura reporta
cuantos casos excluir por datos faltantes.
=====================================================================
"""

# ---------------------------------------------------------------------
# IMPORTACIONES
# ---------------------------------------------------------------------
from pathlib import Path                  # manejo de rutas relativas
import pandas as pd                       # manejo tabular de datos
import matplotlib.pyplot as plt           # motor de graficas
import seaborn as sns                     # graficas estadisticas

# ---------------------------------------------------------------------
# CONFIGURACION DE RUTAS (relativas a la raiz del proyecto)
# ---------------------------------------------------------------------
# __file__ es la ruta de ESTE script; .parent sube UN nivel.
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"                        # carpeta de datos
OUT_DIR  = BASE_DIR / "outputs"                     # carpeta de salidas

# Estilo visual
sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 11})

# ---------------------------------------------------------------------
# CARGA DE DATOS
# ---------------------------------------------------------------------
train_path = DATA_DIR / "train.csv"       # generado por la Etapa 1
if not train_path.exists():
    raise FileNotFoundError(
        "No existe train.csv. Ejecuta primero el script de la Etapa 1."
    )

df_train = pd.read_csv(train_path)        # 200 filas x 9 columnas

# Copia para graficar con etiquetas legibles (el original queda intacto)
df_plot = df_train.copy()
df_plot["Diagnostico"] = df_plot["type"].map(
    {0: "Sin Diabetes (0)", 1: "Con Diabetes (1)"}
)

# =====================================================================
# REVISION DE LA ESTRUCTURA Y RESUMEN DE DATOS
# =====================================================================
print("=" * 60)
print("1. ESTRUCTURA DEL DATASET (ENTRENAMIENTO)")
print("=" * 60)
print(f"Dimensiones: {df_train.shape[0]} filas x "
      f"{df_train.shape[1]} columnas\n")

df_train.info()    # .info() imprime solo; NO envolver en print()

print("\n" + "=" * 60)
print("2. RESUMEN ESTADISTICO (DESCRIPTIVO)")
print("=" * 60)
# count incluye solo casos NO faltantes: comparen contra 200
print(df_train.describe().T
      [["count", "mean", "std", "min", "50%", "max"]]
      .round(2))

# =====================================================================
# GENERACION DE LAS 4 GRAFICAS
# =====================================================================

def reportar_nan(variable):
    """Cuenta cuantos casos quedan fuera de la figura por NaN."""
    n_nan = df_train[variable].isna().sum()
    n_validos = len(df_train) - n_nan
    print(f"  [{variable}] Casos graficados: {n_validos} de "
          f"{len(df_train)} (excluidos por faltantes: {n_nan})")


# ---------- FIGURA 1: Glucosa segun diagnostico ----------
print("\nFIGURA 1: Distribucion de Glucosa por Diagnostico")
reportar_nan("glu")

plt.figure(figsize=(8, 5))
# hue=x + legend=False: evita el FutureWarning de seaborn >= 0.13
sns.boxplot(data=df_plot, x="Diagnostico", y="glu",
            hue="Diagnostico", palette="Set2", legend=False)
# puntos individuales encima del boxplot (alpha=transparencia)
sns.stripplot(data=df_plot, x="Diagnostico", y="glu",
              color="black", alpha=0.3, jitter=0.2)
plt.title("1. Distribucion de Glucosa (glu) segun Resultado",
         fontsize=13, fontweight="bold")
plt.xlabel("Resultado del Diagnostico")
plt.ylabel("Glucosa en Plasma (mg/dL)")
plt.tight_layout()
plt.savefig(OUT_DIR / "act1_01_distribucion_glucosa.png", dpi=150)
plt.show()

# ---------- FIGURA 2: IMC segun diagnostico ----------
print("\nFIGURA 2: Distribucion de IMC por Diagnostico")
reportar_nan("bmi")

plt.figure(figsize=(8, 5))
sns.boxplot(data=df_plot, x="Diagnostico", y="bmi",
            hue="Diagnostico", palette="Set2", legend=False)
sns.stripplot(data=df_plot, x="Diagnostico", y="bmi",
              color="black", alpha=0.3, jitter=0.2)
plt.title("2. Distribucion del IMC (bmi) segun Resultado",
         fontsize=13, fontweight="bold")
plt.xlabel("Resultado del Diagnostico")
plt.ylabel("Indice de Masa Corporal (kg/m^2)")
plt.tight_layout()
plt.savefig(OUT_DIR / "act1_02_distribucion_imc.png", dpi=150)
plt.show()

# ---------- FIGURA 3: Edad segun diagnostico ----------
print("\nFIGURA 3: Distribucion de Edad por Diagnostico")
reportar_nan("age")

plt.figure(figsize=(8, 5))
sns.boxplot(data=df_plot, x="Diagnostico", y="age",
            hue="Diagnostico", palette="Set2", legend=False)
sns.stripplot(data=df_plot, x="Diagnostico", y="age",
              color="black", alpha=0.3, jitter=0.2)
plt.title("3. Distribucion de Edad (age) segun Resultado",
         fontsize=13, fontweight="bold")
plt.xlabel("Resultado del Diagnostico")
plt.ylabel("Edad (anos)")
plt.tight_layout()
plt.savefig(OUT_DIR / "act1_03_distribucion_edad.png", dpi=150)
plt.show()

# ---------- FIGURA 4: Glucosa vs IMC por clase ----------
print("\nFIGURA 4: Glucosa vs IMC por Clase")
print(f"  Casos graficados: {df_train[['glu','bmi']].dropna().shape[0]}"
      f" de {len(df_train)}"
      f" (requieren AMBAS variables validas)")

plt.figure(figsize=(9, 6))
# hue: color por clase | style: simbolo por clase | s: tamano de punto
sns.scatterplot(data=df_plot, x="bmi", y="glu",
                hue="Diagnostico", style="Diagnostico",
                palette="Set1", alpha=0.8, s=70)
plt.title("4. Relacion entre Glucosa e IMC por Clase",
         fontsize=13, fontweight="bold")
plt.xlabel("Indice de Masa Corporal (kg/m^2)")
plt.ylabel("Glucosa en Plasma (mg/dL)")
plt.legend(title="Resultado", loc="upper left")
plt.tight_layout()
plt.savefig(OUT_DIR / "act1_04_relacion_glucosa_imc.png", dpi=150)
plt.show()

print("\n[ETAPA 1 COMPLETADA] Revisa outputs/ para las 4 imagenes")