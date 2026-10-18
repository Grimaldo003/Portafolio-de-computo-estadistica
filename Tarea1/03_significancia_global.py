"""
=====================================================================
ETAPA 3 - SIGNIFICANCIA GLOBAL DEL MODELO
Proyecto: Regresion Logistica - Clasificacion de Diabetes
---------------------------------------------------------------------
OBJETIVO: Contrastar si el conjunto de predictores mejora el modelo
frente a uno que solo tiene intercepto.
---------------------------------------------------------------------
HIPOTESIS
    H0: b1 = b2 = ... = b7 = 0   (ningun predictor aporta)
    H1: al menos un bj != 0      (algun predictor aporta)
---------------------------------------------------------------------
PRUEBAS APLICADAS
    1. Razon de verosimilitud (LR):  LR = 2*(lnL_full - lnL_null)
       Bajo H0, LR ~ chi2 con gl = numero de predictores
    2. Wald: contraste cuadratico sobre los coeficientes y su covarianza
---------------------------------------------------------------------
INPUT : data/train.csv, data/medianas_train.json
OUTPUT: outputs/etapa3_significancia.txt
        outputs/etapa3_chi2.png
=====================================================================
"""

# ---------------------------------------------------------------------
# 1. IMPORTACIONES
# ---------------------------------------------------------------------
from pathlib import Path                  # rutas relativas
import json                               # leer medianas guardadas
import pandas as pd                       # tablas de datos
import numpy as np                        # algebra numerica
import scipy.stats as stats               # distribucion chi-cuadrado
import statsmodels.api as sm              # modelos lineales generalizados
import matplotlib.pyplot as plt           # grafico de la distribucion chi2

# ---------------------------------------------------------------------
# 2. CONFIGURACION
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR  = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

TRAIN_PATH    = DATA_DIR / "train.csv"
MEDIANAS_PATH = DATA_DIR / "medianas_train.json"
VAR_RESPUESTA = "type"
SIGNIFICANCIA = 0.05
PREDICTORES   = ["npreg", "glu", "bp", "skin", "bmi", "ped", "age"]

# ---------------------------------------------------------------------
# 3. CARGA Y RECONSTRUCCION DEL CONJUNTO DE ENTRENAMIENTO
# ---------------------------------------------------------------------
# Se repite la preparacion de la Etapa 2 para que este script sea
# autonomo. Las medianas se LEEN del JSON, no se recalculan: asi se
# garantiza que ambos scripts trabajen exactamente con los mismos datos.
df_train = pd.read_csv(TRAIN_PATH)

with open(MEDIANAS_PATH, encoding="utf-8") as f:
    medianas = json.load(f)               # dict: variable -> mediana

for var, mediana in medianas.items():
    df_train[var] = df_train[var].fillna(mediana)   # imputar con lo guardado

# Verificacion: sin faltantes
assert df_train[PREDICTORES + [VAR_RESPUESTA]].isna().sum().sum() == 0

# ---------------------------------------------------------------------
# 4. DEFINICION DE MATRICES
# ---------------------------------------------------------------------
y     = df_train[VAR_RESPUESTA].values          # vector respuesta (0/1)
X_raw = df_train[PREDICTORES].values            # matriz de predictores
X     = sm.add_constant(X_raw)                  # anadir intercepto
n     = len(y)                                  # tamano de muestra

# =====================================================================
# 5. MODELO NULO (SOLO INTERCEPTO)
# =====================================================================
# El modelo nulo no usa predictores: predice la proporcion base de
# diabetes para todas las observaciones. Sirve como linea base.
print("=" * 64)
print(" 1. MODELO NULO (solo intercepto)")
print("=" * 64)

X_nulo    = np.ones((n, 1))                     # columna de unos
modelo_nulo = sm.Logit(y, X_nulo)               # modelo sin predictores
res_nulo    = modelo_nulo.fit(disp=False)       # ajustar

print(f"Observaciones        : {int(res_nulo.nobs)}")
print(f"Parametros           : {len(res_nulo.params)} (solo intercepto)")
print(f"Log-verosimilitud    : {res_nulo.llf:.4f}")
print(f"AIC                  : {res_nulo.aic:.2f}")
print(f"BIC                  : {res_nulo.bic:.2f}")

# =====================================================================
# 6. MODELO COMPLETO
# =====================================================================
print("\n" + "=" * 64)
print(" 2. MODELO COMPLETO (7 predictores)")
print("=" * 64)

modelo_full = sm.Logit(y, X)
res_full    = modelo_full.fit(disp=False)

print(f"Observaciones        : {int(res_full.nobs)}")
print(f"Parametros           : {len(res_full.params)} (intercepto + 7)")
print(f"Log-verosimilitud    : {res_full.llf:.4f}")
print(f"AIC                  : {res_full.aic:.2f}")
print(f"BIC                  : {res_full.bic:.2f}")

# =====================================================================
# 7. PRUEBA DE RAZON DE VEROSIMILITUD (LR)
# =====================================================================
# LR = 2 * (lnL_completo - lnL_nulo)
# Bajo H0, LR ~ chi2 con gl = numero de restricciones = 7
print("\n" + "=" * 64)
print(" 3. PRUEBA DE RAZON DE VEROSIMILITUD (LR)")
print("=" * 64)

ll_full = res_full.llf                          # log-verosimilitud completa
ll_nulo = res_nulo.llf                          # log-verosimilitud nula

lr_estadistico = 2 * (ll_full - ll_nulo)        # estadistico LR
gl             = len(PREDICTORES)               # grados de libertad = 7
p_lr           = stats.chi2.sf(lr_estadistico, gl)   # p-valor (cola derecha)
valor_critico  = stats.chi2.ppf(1 - SIGNIFICANCIA, gl)  # critico al 5%

print(f"lnL modelo completo  : {ll_full:.4f}")
print(f"lnL modelo nulo      : {ll_nulo:.4f}")
print(f"Diferencia (2*delta) : {lr_estadistico:.4f}")
print(f"Grados de libertad   : {gl}")
print(f"Valor critico (5%)   : {valor_critico:.4f}")
print(f"p-valor              : {p_lr:.3e}")

# Verificacion cruzada: statsmodels ya calcula el LR internamente
print(f"\n[VERIFICACION] res_full.llr        = {res_full.llr:.4f}")
print(f"[VERIFICACION] res_full.llr_pvalue = {res_full.llr_pvalue:.3e}")
print(f"[VERIFICACION] gl de statsmodels   = {int(res_full.df_model)}")

# Decision estadistica
if p_lr < SIGNIFICANCIA:
    decision = "RECHAZAR H0"
    conclusion = ("El conjunto de predictores mejora significativamente "
                  "el modelo frente al de solo intercepto.")
else:
    decision = "NO RECHAZAR H0"
    conclusion = ("No hay evidencia suficiente para afirmar que los "
                  "predictores aportan al modelo.")

print(f"\nDECISION: {decision}")
print(f"CONCLUSION: {conclusion}")

# =====================================================================
# 8. PRUEBA DE WALD (COMPLEMENTARIA)
# =====================================================================
# W = b' * V^-1 * b, donde b son los coeficientes (sin intercepto) y
# V su matriz de covarianza. Bajo H0, W ~ chi2 con gl = 7.
print("\n" + "=" * 64)
print(" 4. PRUEBA DE WALD (complementaria)")
print("=" * 64)

coef_todos = np.asarray(res_full.params)        # incluye intercepto
cov_todos  = np.asarray(res_full.cov_params())  # matriz de covarianza

coef_slopes = coef_todos[1:]                    # excluir intercepto
cov_slopes  = cov_todos[1:, 1:]                 # submatriz sin intercepto

# Forma cuadratica b' V^-1 b
wald_estadistico = float(coef_slopes @ np.linalg.inv(cov_slopes) @ coef_slopes)
p_wald           = stats.chi2.sf(wald_estadistico, gl)

print(f"Estadistico Wald     : {wald_estadistico:.4f}")
print(f"Grados de libertad   : {gl}")
print(f"p-valor              : {p_wald:.3e}")
print("\nNota: LR y Wald son asintoticamente equivalentes. Diferencias")
print("grandes entre ambos sugieren muestras pequenas o efectos extremos.")

# =====================================================================
# 9. COMPARACION DE MODELOS
# =====================================================================
print("\n" + "=" * 64)
print(" 5. COMPARACION NULO vs COMPLETO")
print("=" * 64)

comparacion = pd.DataFrame({
    "Modelo":            ["Nulo (intercepto)", "Completo (7 predictores)"],
    "Parametros":        [1, 8],
    "Log_Verosimilitud": [ll_nulo, ll_full],
    "AIC":               [res_nulo.aic, res_full.aic],
    "BIC":               [res_nulo.bic, res_full.bic],
    "Pseudo_R2_McFadden":[0.0, res_full.prsquared],
})
print(comparacion.to_string(index=False))

print(f"\nReduccion de AIC : {res_nulo.aic - res_full.aic:.2f}")
print(f"Reduccion de BIC : {res_nulo.bic - res_full.bic:.2f}")
print("(Menor AIC/BIC indica mejor equilibrio entre ajuste y complejidad)")

# =====================================================================
# 10. GRAFICO DE LA DISTRIBUCION CHI-CUADRADO
# =====================================================================
# Se visualiza donde cae el estadistico observado respecto a la
# distribucion teorica bajo H0. Si cae muy a la derecha del valor
# critico, la evidencia contra H0 es fuerte.
x = np.linspace(0, max(lr_estadistico * 1.15, valor_critico * 3), 900)
densidad = stats.chi2.pdf(x, gl)                # densidad chi2 con gl=7

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x, densidad, color="steelblue", linewidth=2,
        label=f"chi2 con {gl} grados de libertad")
ax.fill_between(x, densidad, where=(x >= valor_critico),
                color="orange", alpha=0.35,
                label=f"Region de rechazo (5%)")
ax.axvline(valor_critico, color="darkorange", linestyle=":", linewidth=2,
           label=f"Valor critico = {valor_critico:.2f}")
ax.axvline(lr_estadistico, color="crimson", linestyle="--", linewidth=2,
           label=f"LR observado = {lr_estadistico:.2f}")
ax.set_title("Prueba de razon de verosimilitud: distribucion bajo H0",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Estadistico LR")
ax.set_ylabel("Densidad")
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig(OUT_DIR / "etapa3_chi2.png", dpi=150)
plt.show()

# =====================================================================
# 11. GUARDAR RESULTADOS
# =====================================================================
with open(OUT_DIR / "etapa3_significancia.txt", "w", encoding="utf-8") as f:
    f.write("SIGNIFICANCIA GLOBAL DEL MODELO\n")
    f.write("=" * 56 + "\n\n")
    f.write("H0: b1 = b2 = ... = b7 = 0 (ningun predictor aporta)\n")
    f.write("H1: al menos un bj != 0\n\n")
    f.write(f"lnL modelo completo : {ll_full:.4f}\n")
    f.write(f"lnL modelo nulo     : {ll_nulo:.4f}\n")
    f.write(f"Estadistico LR      : {lr_estadistico:.4f}\n")
    f.write(f"Grados de libertad  : {gl}\n")
    f.write(f"Valor critico (5%)  : {valor_critico:.4f}\n")
    f.write(f"p-valor (LR)        : {p_lr:.3e}\n")
    f.write(f"Estadistico Wald    : {wald_estadistico:.4f}\n")
    f.write(f"p-valor (Wald)      : {p_wald:.3e}\n\n")
    f.write(f"Decision: {decision}\n")
    f.write(f"{conclusion}\n\n")
    f.write(comparacion.to_string(index=False))
    f.write("\n")

print("\n[GUARDADO] outputs/etapa3_significancia.txt")
print("[GUARDADO] outputs/etapa3_chi2.png")

print("\n" + "=" * 64)
print(" ETAPA 3 COMPLETADA")
print("=" * 64)
print("PROXIMO PASO: Etapa 4 - modelo reducido (step / AIC / clinica)")