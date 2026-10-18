"""
=====================================================================
ETAPA 5 - PREDICCION SOBRE EL CONJUNTO DE PRUEBA
(seccion 6 del documento de tarea)
Proyecto: Regresion Logistica - Clasificacion de Diabetes
---------------------------------------------------------------------
OBJETIVO: Aplicar los modelos ajustados con train.csv sobre test.csv,
calcular probabilidades predichas y clasificar con umbral 0.50.
---------------------------------------------------------------------
INPUT : data/train.csv
        data/test.csv
        data/medianas_train.json
OUTPUT: outputs/etapa5_predicciones.csv
        outputs/etapa5_distribucion_probabilidades.png
        outputs/etapa5_resumen.txt
---------------------------------------------------------------------
REGLA ANTI-FUGA DE INFORMACION
    Las medianas para imputar el conjunto de prueba se LEEN del JSON
    generado en la Etapa 2 (calculadas solo con train). Nunca se
    recalculan con test.csv. Recalcularlas introduciria informacion
    del conjunto de evaluacion dentro del preprocesamiento.
---------------------------------------------------------------------
MODELOS EVALUADOS (los tres de la Etapa 4)
    Completo : npreg, glu, bp, skin, bmi, ped, age
    Reducido : npreg, glu, bmi
    Clinico  : npreg, glu, bmi, ped          <- modelo principal
---------------------------------------------------------------------
FORMULAS
    z_i = b0 + b1*x1i + ... + bk*xki         (predictor lineal)
    p_i = 1 / (1 + exp(-z_i))                (funcion logistica)
    clase_i = 1 si p_i >= 0.50, 0 en otro caso
=====================================================================
"""

# ---------------------------------------------------------------------
# 1. IMPORTACIONES
# ---------------------------------------------------------------------
from pathlib import Path                  # rutas relativas al proyecto
import json                               # lectura de las medianas
import numpy as np                        # operaciones numericas
import pandas as pd                       # tablas de datos
import statsmodels.api as sm              # regresion logistica
import matplotlib.pyplot as plt           # graficos

# ---------------------------------------------------------------------
# 2. CONFIGURACION
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent   # Tarea1/
DATA_DIR = BASE_DIR / "data"                 # carpeta de datos
OUT_DIR  = BASE_DIR / "outputs"              # carpeta de salidas
OUT_DIR.mkdir(exist_ok=True)

TRAIN_PATH    = DATA_DIR / "train.csv"       # 200 observaciones
TEST_PATH     = DATA_DIR / "test.csv"        # 332 observaciones
MEDIANAS_PATH = DATA_DIR / "medianas_train.json"

VAR_RESPUESTA       = "type"                 # variable binaria objetivo
VARIABLE_DESCARTADA = "insulin"              # 50% faltantes: se excluye
UMBRAL              = 0.50                   # umbral de clasificacion

# Conjuntos de predictores de cada modelo (Etapa 4)
MODELOS = {
    "Completo": ["npreg", "glu", "bp", "skin", "bmi", "ped", "age"],
    "Reducido": ["npreg", "glu", "bmi"],
    "Clinico":  ["npreg", "glu", "bmi", "ped"],
}

# Modelo que se usara como principal en las etapas siguientes
MODELO_PRINCIPAL = "Clinico"

# =====================================================================
# 3. CARGA DE DATOS Y MEDIANAS
# =====================================================================
print("=" * 66)
print(" 1. CARGA DE DATOS")
print("=" * 66)

df_train = pd.read_csv(TRAIN_PATH)           # conjunto de ajuste
df_test  = pd.read_csv(TEST_PATH)            # conjunto de evaluacion

with open(MEDIANAS_PATH, encoding="utf-8") as f:
    medianas = json.load(f)                  # dict: variable -> mediana

print(f"Train : {df_train.shape[0]} filas x {df_train.shape[1]} columnas")
print(f"Test  : {df_test.shape[0]} filas x {df_test.shape[1]} columnas")
print(f"Medianas recuperadas de train: {medianas}")

# =====================================================================
# 4. PREPARACION DE AMBOS CONJUNTOS
# =====================================================================
def preparar(df, medianas):
    """
    Aplica a un conjunto el mismo preprocesamiento de la Etapa 2:
      1. Elimina la variable 'insulin' (50% de faltantes).
      2. Imputa los faltantes restantes con las medianas de TRAIN.

    El parametro 'medianas' siempre proviene del entrenamiento, tanto
    para train como para test. Asi no hay fuga de informacion.
    """
    df = df.drop(columns=[VARIABLE_DESCARTADA])       # quitar insulin
    for var, valor in medianas.items():               # imputar faltantes
        df[var] = df[var].fillna(valor)
    return df

df_train_prep = preparar(df_train.copy(), medianas)
df_test_prep  = preparar(df_test.copy(), medianas)

# Verificacion: ningun faltante pendiente en test
faltantes_test = df_test_prep.isna().sum().sum()
print(f"\nFaltantes pendientes en test tras imputacion: {faltantes_test}")
assert faltantes_test == 0, "Quedan faltantes sin tratar en test.csv"

# Distribucion real de clases en el conjunto de prueba
y_test = df_test_prep[VAR_RESPUESTA].values.astype(int)
n_pos = int(y_test.sum())
n_neg = int(len(y_test) - n_pos)
print(f"Clases reales en test: 0 -> {n_neg} | 1 -> {n_pos} "
      f"(prevalencia {100*n_pos/len(y_test):.1f}%)")

# =====================================================================
# 5. AJUSTE EN TRAIN Y PREDICCION EN TEST
# =====================================================================
# Los modelos se reajustan aqui sobre train para que este script sea
# autocontenido. La prediccion se hace sobre test, que el modelo nunca vio.
print("\n" + "=" * 66)
print(" 2. AJUSTE Y PREDICCION")
print("=" * 66)

y_train = df_train_prep[VAR_RESPUESTA].values.astype(int)

resultados = {}          # nombre -> dict con modelo, probs, clases

for nombre, variables in MODELOS.items():
    # --- Ajuste sobre entrenamiento ---
    # Se pasa un DataFrame (no .values) para conservar los nombres.
    # El orden de columnas debe ser identico en train y test: al
    # construir ambos con la misma lista 'variables', queda garantizado.
    X_train = sm.add_constant(df_train_prep[variables])
    modelo  = sm.Logit(y_train, X_train).fit(disp=False)

    # --- Prediccion sobre prueba ---
    X_test = sm.add_constant(df_test_prep[variables])
    probs  = modelo.predict(X_test)              # probabilidades p_i
    clases = (probs >= UMBRAL).astype(int)       # clasificacion a 0.50

    # --- Resumen rapido ---
    n_pred_pos = int(clases.sum())
    print(f"\n[{nombre}] {len(variables)} predictores")
    print(f"  Variables      : {', '.join(variables)}")
    print(f"  AIC (train)    : {modelo.aic:.2f}")
    print(f"  p_min / p_max  : {probs.min():.4f} / {probs.max():.4f}")
    print(f"  p_media        : {probs.mean():.4f}")
    print(f"  Positivos pred.: {n_pred_pos} de {len(clases)} "
          f"({100*n_pred_pos/len(clases):.1f}%)")

    resultados[nombre] = {
        "modelo":    modelo,
        "probs":     np.asarray(probs),
        "clases":    clases,
        "variables": variables,
    }

# =====================================================================
# 6. MATRIZ DE CONFUSION (PREVIA)
# =====================================================================
# Aqui solo se muestran los conteos basicos para verificar que la
# clasificacion funciona. Las metricas completas (exactitud,
# sensibilidad, especificidad, F1, AUC) corresponden a la etapa siguiente.
print("\n" + "=" * 66)
print(" 3. CONTEO BASICO DE ACIERTOS (umbral 0.50)")
print("=" * 66)

for nombre, datos in resultados.items():
    clases = datos["clases"]
    # Verdaderos positivos, falsos positivos, falsos negativos, verdaderos negativos
    tp = int(((clases == 1) & (y_test == 1)).sum())   # predijo 1, era 1
    fp = int(((clases == 1) & (y_test == 0)).sum())   # predijo 1, era 0
    fn = int(((clases == 0) & (y_test == 1)).sum())   # predijo 0, era 1
    tn = int(((clases == 0) & (y_test == 0)).sum())   # predijo 0, era 0
    aciertos = tp + tn
    print(f"\n[{nombre}]")
    print(f"  TP={tp:4d}  FP={fp:4d}")
    print(f"  FN={fn:4d}  TN={tn:4d}")
    print(f"  Aciertos totales: {aciertos} de {len(clases)} "
          f"({100*aciertos/len(clases):.1f}%)")

# =====================================================================
# 7. DISTRIBUCION DE PROBABILIDADES PREDICHAS
# =====================================================================
# Se separa por clase real: si el modelo discrimina, los casos con
# diabetes deberian concentrarse a la derecha del umbral.
print("\n" + "=" * 66)
print(" 4. GRAFICO DE DISTRIBUCION")
print("=" * 66)

probs_principal = resultados[MODELO_PRINCIPAL]["probs"]

fig, ax = plt.subplots(figsize=(10, 5))
# Histograma de los casos SIN diabetes (clase real 0)
ax.hist(probs_principal[y_test == 0], bins=20, alpha=0.6,
        label="Sin diabetes (real 0)", color="steelblue")
# Histograma de los casos CON diabetes (clase real 1)
ax.hist(probs_principal[y_test == 1], bins=20, alpha=0.6,
        label="Con diabetes (real 1)", color="crimson")
# Linea vertical en el umbral de decision
ax.axvline(UMBRAL, color="black", linestyle="--", linewidth=2,
           label=f"Umbral = {UMBRAL:.2f}")
ax.set_title(f"Distribucion de probabilidades predichas - modelo {MODELO_PRINCIPAL}",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Probabilidad predicha de diabetes")
ax.set_ylabel("Numero de pacientes")
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "etapa5_distribucion_probabilidades.png", dpi=150)
plt.show()

# =====================================================================
# 8. GUARDAR PREDICCIONES
# =====================================================================
# Se guarda una tabla con la clase real y, por cada modelo, la
# probabilidad y la clase predicha. Las etapas 6, 7 y 8 consumen
# este archivo en lugar de reajustar los modelos.
salida = pd.DataFrame({"y_real": y_test})

for nombre, datos in resultados.items():
    salida[f"p_{nombre}"]     = datos["probs"]                  # probabilidad
    salida[f"clase_{nombre}"] = datos["clases"]                 # clase a 0.50

salida.to_csv(OUT_DIR / "etapa5_predicciones.csv", index=False)
print(f"\n[GUARDADO] outputs/etapa5_predicciones.csv "
      f"({salida.shape[0]} filas x {salida.shape[1]} columnas)")

# Vista previa de las primeras filas
print("\nPrimeras 10 predicciones:")
print(salida.head(10).round(4).to_string(index=False))

# =====================================================================
# 9. RESUMEN EN TEXTO
# =====================================================================
with open(OUT_DIR / "etapa5_resumen.txt", "w", encoding="utf-8") as f:
    f.write("PREDICCION SOBRE EL CONJUNTO DE PRUEBA\n")
    f.write("=" * 58 + "\n\n")
    f.write(f"Umbral de clasificacion : {UMBRAL}\n")
    f.write(f"Observaciones en test   : {len(y_test)}\n")
    f.write(f"Prevalencia real        : {100*n_pos/len(y_test):.1f}%\n")
    f.write(f"Modelo principal        : {MODELO_PRINCIPAL}\n\n")

    for nombre, datos in resultados.items():
        clases = datos["clases"]
        tp = int(((clases == 1) & (y_test == 1)).sum())
        fp = int(((clases == 1) & (y_test == 0)).sum())
        fn = int(((clases == 0) & (y_test == 1)).sum())
        tn = int(((clases == 0) & (y_test == 0)).sum())
        f.write(f"[{nombre}] variables: {', '.join(datos['variables'])}\n")
        f.write(f"  Matriz de confusion: TP={tp} FP={fp} FN={fn} TN={tn}\n")
        f.write(f"  Aciertos: {tp+tn}/{len(clases)} "
                f"({100*(tp+tn)/len(clases):.1f}%)\n\n")

    f.write("Nota: las metricas completas (exactitud, sensibilidad,\n")
    f.write("especificidad, precision, VPN, F1, exactitud balanceada y\n")
    f.write("AUC) se calculan en la etapa siguiente.\n")

print("[GUARDADO] outputs/etapa5_resumen.txt")

print("\n" + "=" * 66)
print(" ETAPA 5 COMPLETADA")
print("=" * 66)
print("PROXIMO PASO: Etapa 6 - metricas de desempeno y curva ROC")