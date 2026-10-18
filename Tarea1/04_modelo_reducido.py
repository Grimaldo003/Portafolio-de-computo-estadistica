"""
=====================================================================
ETAPA 4 - CONSTRUCCION DE UN MODELO REDUCIDO
Proyecto: Regresion Logistica - Clasificacion de Diabetes
---------------------------------------------------------------------
OBJETIVO: Construir y comparar modelos reducidos bajo tres criterios:
significancia estadistica, AIC/BIC y juicio clinico.
---------------------------------------------------------------------
INPUT : data/train.csv, data/medianas_train.json
OUTPUT: outputs/etapa4_modelo_reducido.txt
        outputs/etapa4_comparacion.txt
        outputs/etapa4_aic_evolution.png
---------------------------------------------------------------------
CORRECCION RESPECTO A LA VERSION ANTERIOR
  1. Se pasa un DataFrame de pandas a sm.Logit (no .values). Asi
     statsmodels conserva los nombres de las variables y pvalues
     se devuelve como Series con indice, no como ndarray.
  2. Se corrige la logica del bucle stepwise: se elimina mientras
     el p-valor maximo sea >= alpha (antes estaba invertido).
  3. El historial de AIC se registra en cada paso para graficarlo.
=====================================================================
"""

# ---------------------------------------------------------------------
# 1. IMPORTACIONES
# ---------------------------------------------------------------------
from pathlib import Path                  # rutas relativas al proyecto
import json                               # lectura de las medianas
import pandas as pd                       # tablas de datos
import numpy as np                        # operaciones numericas
import scipy.stats as stats               # distribucion chi-cuadrado
import statsmodels.api as sm              # regresion logistica
import matplotlib.pyplot as plt           # graficos

# ---------------------------------------------------------------------
# 2. CONFIGURACION
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent   # Tarea1/
DATA_DIR = BASE_DIR / "data"                 # datos
OUT_DIR  = BASE_DIR / "outputs"              # salidas
OUT_DIR.mkdir(exist_ok=True)

TRAIN_PATH    = DATA_DIR / "train.csv"
MEDIANAS_PATH = DATA_DIR / "medianas_train.json"
VAR_RESPUESTA = "type"                       # variable binaria objetivo
SIGNIFICANCIA = 0.05                         # umbral alfa para stepwise

# =====================================================================
# 3. CARGA Y PREPARACION
# =====================================================================
df_train = pd.read_csv(TRAIN_PATH)           # 200 filas x 9 columnas

with open(MEDIANAS_PATH, encoding="utf-8") as f:
    medianas = json.load(f)                  # dict: variable -> mediana

# Imputamos con las medianas guardadas en la Etapa 2 (no se recalculan)
for var, mediana in medianas.items():
    df_train[var] = df_train[var].fillna(mediana)

print("=" * 66)
print(" 1. CARGA DE DATOS")
print("=" * 66)
print(f"Observaciones: {len(df_train)}")

# =====================================================================
# 4. FUNCIONES AUXILIARES
# =====================================================================

def ajustar(df, variables, y):
    """
    Ajusta un modelo logit usando un DataFrame de pandas.
    Al pasar un DataFrame (no un ndarray), statsmodels conserva los
    nombres de las columnas y devuelve params/pvalues como Series.
    """
    X = sm.add_constant(df[variables])       # anade columna 'const'
    return sm.Logit(y, X).fit(disp=False)    # estima por maxima verosimilitud


def tabla_coef(res, variables):
    """
    Extrae los resultados a un DataFrame etiquetado.
    Se construyen los nombres manualmente para no depender del tipo
    que devuelva statsmodels (Series o ndarray).
    """
    nombres = ["const"] + list(variables)    # orden: intercepto primero
    return pd.DataFrame({
        "Variable": nombres,
        "Coef":     np.asarray(res.params),   # coeficientes beta
        "SE":       np.asarray(res.bse),      # errores estandar
        "z":        np.asarray(res.tvalues),  # estadistico z
        "p_valor":  np.asarray(res.pvalues),  # p-valor
        "OR":       np.exp(np.asarray(res.params)),  # odds ratio
    })

# =====================================================================
# 5. MODELO COMPLETO (LINEA BASE)
# =====================================================================
print("\n" + "=" * 66)
print(" 2. MODELO COMPLETO (7 predictores)")
print("=" * 66)

PRED_COMPLETO = ["npreg", "glu", "bp", "skin", "bmi", "ped", "age"]
y = df_train[VAR_RESPUESTA].values           # vector respuesta

res_full = ajustar(df_train, PRED_COMPLETO, y)
tab_full = tabla_coef(res_full, PRED_COMPLETO)

print(tab_full.round(4).to_string(index=False))
print(f"\nAIC: {res_full.aic:.2f} | BIC: {res_full.bic:.2f} "
      f"| Pseudo-R2: {res_full.prsquared:.4f}")

# =====================================================================
# 6. STEPWISE BACKWARD
# =====================================================================
print("\n" + "=" * 66)
print(" 3. STEPWISE BACKWARD (eliminacion por p-valor)")
print("=" * 66)

def stepwise_backward(df, y, variables, alpha=0.05):
    """
    Elimina iterativamente el predictor con mayor p-valor mientras
    ese p-valor sea >= alpha. Se detiene cuando todos los predictores
    restantes tienen p < alpha, o cuando queda uno solo.

    Devuelve: (lista de variables retenidas, resultado final, historial)
    """
    actuales = list(variables)               # copia mutable
    historial = []                           # registro de cada paso

    while True:
        res = ajustar(df, actuales, y)       # ajustar modelo actual
        tab = tabla_coef(res, actuales)      # tabla etiquetada
        pred = tab[tab["Variable"] != "const"]   # solo predictores

        # Identificar el predictor con el p-valor mas alto
        idx_peor = pred["p_valor"].idxmax()  # posicion del maximo
        peor_var = pred.loc[idx_peor, "Variable"]
        peor_p   = pred.loc[idx_peor, "p_valor"]

        # Registrar el estado de este modelo
        historial.append({
            "n_vars":    len(actuales),
            "variables": list(actuales),
            "AIC":       res.aic,
            "BIC":       res.bic,
            "peor_var":  peor_var,
            "peor_p":    peor_p,
        })

        print(f"  [{len(actuales)} vars] AIC={res.aic:7.2f} | "
              f"peor: {peor_var} (p={peor_p:.4f})")

        # Condicion de parada: todos significativos o queda uno solo
        if peor_p < alpha or len(actuales) == 1:
            return actuales, res, historial

        actuales.remove(peor_var)            # eliminar y repetir

PRED_REDUCIDO, res_reducido, historial = stepwise_backward(
    df_train, y, PRED_COMPLETO, alpha=SIGNIFICANCIA
)

print(f"\n[RESULTADO] Variables retenidas: {PRED_REDUCIDO}")
print(tabla_coef(res_reducido, PRED_REDUCIDO).round(4).to_string(index=False))
print(f"\nAIC: {res_reducido.aic:.2f} | BIC: {res_reducido.bic:.2f} "
      f"| Pseudo-R2: {res_reducido.prsquared:.4f}")

# =====================================================================
# 7. PRUEBA LR: COMPLETO vs REDUCIDO
# =====================================================================
# LR = 2*(lnL_completo - lnL_reducido) ~ chi2 con gl = variables quitadas
print("\n" + "=" * 66)
print(" 4. PRUEBA DE RAZON DE VEROSIMILITUD (completo vs reducido)")
print("=" * 66)

ll_full = res_full.llf
ll_red  = res_reducido.llf
lr_stat = 2 * (ll_full - ll_red)             # estadistico LR
gl_diff = len(PRED_COMPLETO) - len(PRED_REDUCIDO)   # grados de libertad
p_lr    = stats.chi2.sf(lr_stat, gl_diff)    # p-valor (cola derecha)

print(f"logLik completo : {ll_full:.4f}")
print(f"logLik reducido : {ll_red:.4f}")
print(f"LR              : {lr_stat:.4f}")
print(f"Grados libertad : {gl_diff}")
print(f"p-valor         : {p_lr:.4f}")

if p_lr < SIGNIFICANCIA:
    conclusion_lr = ("El modelo reducido PIERDE capacidad de ajuste de forma "
                     "significativa. Conviene conservar el modelo completo.")
else:
    conclusion_lr = ("El modelo reducido NO pierde capacidad de ajuste de "
                     "forma significativa. Es preferible por parsimonia.")
print(f"\nConclusion: {conclusion_lr}")

# =====================================================================
# 8. MODELO CON CRITERIO CLINICO
# =====================================================================
# Se retiene 'ped' (antecedentes familiares) aunque su p-valor pueda
# superar 0.05, porque es un factor de riesgo medico establecido.
print("\n" + "=" * 66)
print(" 5. MODELO CON CRITERIO CLINICO (retencion de 'ped')")
print("=" * 66)

PRED_CLINICO = list(PRED_REDUCIDO)           # partir del reducido
if "ped" not in PRED_CLINICO:
    PRED_CLINICO.append("ped")               # forzar inclusion
    print("Se reincorpora 'ped' por relevancia clinica.")

res_clinico = ajustar(df_train, PRED_CLINICO, y)
print(tabla_coef(res_clinico, PRED_CLINICO).round(4).to_string(index=False))
print(f"\nAIC: {res_clinico.aic:.2f} | BIC: {res_clinico.bic:.2f} "
      f"| Pseudo-R2: {res_clinico.prsquared:.4f}")

# =====================================================================
# 9. COMPARACION FINAL
# =====================================================================
print("\n" + "=" * 66)
print(" 6. COMPARACION DE MODELOS")
print("=" * 66)

tabla_cmp = pd.DataFrame({
    "Modelo":        ["Completo", "Reducido", "Clinico"],
    "N_pred":        [len(PRED_COMPLETO), len(PRED_REDUCIDO), len(PRED_CLINICO)],
    "Variables":     [",".join(PRED_COMPLETO),
                      ",".join(PRED_REDUCIDO),
                      ",".join(PRED_CLINICO)],
    "LogLik":        [ll_full, ll_red, res_clinico.llf],
    "AIC":           [res_full.aic, res_reducido.aic, res_clinico.aic],
    "BIC":           [res_full.bic, res_reducido.bic, res_clinico.bic],
    "Pseudo_R2":     [res_full.prsquared, res_reducido.prsquared,
                      res_clinico.prsquared],
})
print(tabla_cmp.round(4).to_string(index=False))

# Eleccion por AIC
modelos = {
    "Completo": res_full.aic,
    "Reducido": res_reducido.aic,
    "Clinico":  res_clinico.aic,
}
elegido = min(modelos, key=modelos.get)
print(f"\nMenor AIC: {elegido} ({modelos[elegido]:.2f})")

# =====================================================================
# 10. GRAFICO: EVOLUCION DEL AIC
# =====================================================================
# Cada entrada del historial es un modelo ajustado durante el stepwise.
# Se grafica el AIC frente al numero de predictores de ese modelo.
n_vars_hist = [h["n_vars"] for h in historial]
aic_hist    = [h["AIC"]    for h in historial]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(n_vars_hist, aic_hist, marker="o", linewidth=2, markersize=8,
        color="steelblue")
ax.set_xlabel("Numero de predictores en el modelo")
ax.set_ylabel("AIC")
ax.set_title("Evolucion del AIC durante el stepwise backward",
             fontsize=13, fontweight="bold")
ax.axhline(min(aic_hist), color="crimson", linestyle="--",
           label=f"AIC minimo = {min(aic_hist):.2f}")
ax.invert_xaxis()                            # de mas a menos predictores
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "etapa4_aic_evolution.png", dpi=150)
plt.show()

# =====================================================================
# 11. GUARDAR RESULTADOS
# =====================================================================
with open(OUT_DIR / "etapa4_modelo_reducido.txt", "w", encoding="utf-8") as f:
    f.write("MODELO REDUCIDO - RESULTADOS\n")
    f.write("=" * 58 + "\n\n")

    f.write("Modelo completo:\n")
    f.write(tab_full.round(4).to_string(index=False))
    f.write(f"\nAIC: {res_full.aic:.2f} | BIC: {res_full.bic:.2f} | "
            f"Pseudo-R2: {res_full.prsquared:.4f}\n\n")

    f.write("Modelo reducido (stepwise backward):\n")
    f.write(f"Variables: {', '.join(PRED_REDUCIDO)}\n")
    f.write(tabla_coef(res_reducido, PRED_REDUCIDO).round(4).to_string(index=False))
    f.write(f"\nAIC: {res_reducido.aic:.2f} | BIC: {res_reducido.bic:.2f} | "
            f"Pseudo-R2: {res_reducido.prsquared:.4f}\n\n")

    f.write("Modelo clinico:\n")
    f.write(f"Variables: {', '.join(PRED_CLINICO)}\n")
    f.write(f"AIC: {res_clinico.aic:.2f} | BIC: {res_clinico.bic:.2f} | "
            f"Pseudo-R2: {res_clinico.prsquared:.4f}\n\n")

    f.write("Prueba LR (completo vs reducido):\n")
    f.write(f"  LR = {lr_stat:.4f} | gl = {gl_diff} | p = {p_lr:.4f}\n")
    f.write(f"  {conclusion_lr}\n\n")

    f.write("Trazado del stepwise:\n")
    for h in historial:
        f.write(f"  {h['n_vars']} vars | AIC={h['AIC']:.2f} | "
                f"BIC={h['BIC']:.2f} | peor={h['peor_var']} "
                f"(p={h['peor_p']:.4f})\n")

with open(OUT_DIR / "etapa4_comparacion.txt", "w", encoding="utf-8") as f:
    f.write("COMPARACION DE MODELOS\n")
    f.write("=" * 58 + "\n\n")
    f.write(tabla_cmp.round(4).to_string(index=False))
    f.write(f"\n\nModelo con menor AIC: {elegido}\n")

print("\n[GUARDADO] outputs/etapa4_modelo_reducido.txt")
print("[GUARDADO] outputs/etapa4_comparacion.txt")
print("[GUARDADO] outputs/etapa4_aic_evolution.png")

# =====================================================================
# 12. RESPUESTAS A LAS PREGUNTAS DE LA TAREA
# =====================================================================
print("\n" + "=" * 66)
print(" 7. RESPUESTAS A LAS PREGUNTAS DE LA TAREA")
print("=" * 66)

eliminadas = [v for v in PRED_COMPLETO if v not in PRED_REDUCIDO]

print("\nP1) Variables eliminadas:")
print(f"    {', '.join(eliminadas) if eliminadas else 'ninguna'}")

print("\nP2) Perdida significativa de ajuste:")
print(f"    p-valor LR = {p_lr:.4f}")
print(f"    {'Si, hay perdida significativa.' if p_lr < SIGNIFICANCIA else 'No, la perdida no es significativa.'}")

print("\nP3) Modelo elegido:")
print(f"    {elegido} (AIC = {modelos[elegido]:.2f})")

print("\nP4) Coincidencia con las graficas exploratorias:")
print("    glu y bmi separan claramente las clases en la Actividad 1")
print("    y quedan retenidas en todos los modelos. bp, skin y age")
print("    mostraron amplio traslape y resultan no significativas.")

print("\n" + "=" * 66)
print(" ETAPA 4 COMPLETADA")
print("=" * 66)
print("PROXIMO PASO: Etapa 5 - prediccion en datos de prueba")