"""
=====================================================================
ETAPA 2 - MODELO LOGISTICO COMPLETO
Proyecto: Regresion Logistica - Clasificacion de Diabetes
---------------------------------------------------------------------
OBJETIVO: Ajustar el modelo con todos los predictores utilizables y
reportar coeficientes, errores estandar, valores p, odds ratios e IC 95%.
---------------------------------------------------------------------
INPUT : data/train.csv
OUTPUT: outputs/etapa2_coeficientes.txt
        outputs/etapa2_interpretaciones.txt
        data/medianas_train.json
=====================================================================
"""

# ---------------------------------------------------------------------
# 1. IMPORTACIONES
# ---------------------------------------------------------------------
from pathlib import Path
import json
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm

# ---------------------------------------------------------------------
# 2. CONFIGURACION
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR  = BASE_DIR / "outputs"
DATA_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

TRAIN_PATH           = DATA_DIR / "train.csv"
VAR_RESPUESTA        = "type"
SIGNIFICANCIA        = 0.05
VARIABLE_DESCARTADA  = "insulin"
PREDICTORES          = ["npreg", "glu", "bp", "skin", "bmi", "ped", "age"]

# ---------------------------------------------------------------------
# 3. CARGA
# ---------------------------------------------------------------------
if not TRAIN_PATH.exists():
    raise FileNotFoundError("No existe train.csv. Ejecuta primero 'tarea.py'.")

df_train = pd.read_csv(TRAIN_PATH)

# =====================================================================
# 4. DIAGNOSTICO DE VALORES FALTANTES
# =====================================================================
print("=" * 62)
print(" 1. DIAGNOSTICO DE VALORES FALTANTES (entrenamiento)")
print("=" * 62)

n_total = len(df_train)
print(f"Filas originales: {n_total}\n")
print(f"  {'Variable':<10}{'Faltantes':>10}{'Porcentaje':>13}")

for var in ["npreg", "glu", "bp", "skin", "insulin", "bmi", "ped", "age"]:
    n_falt = int(df_train[var].isna().sum())
    pct    = 100 * n_falt / n_total
    print(f"  {var:<10}{n_falt:>10}{pct:>12.1f}%")

# =====================================================================
# 5. TRATAMIENTO DE FALTANTES
# =====================================================================
print("\n" + "=" * 62)
print(" 2. TRATAMIENTO DE VALORES FALTANTES")
print("=" * 62)

df_modelo = df_train.drop(columns=[VARIABLE_DESCARTADA])
print(f"[DESCARTADA] '{VARIABLE_DESCARTADA}': "
      f"{df_train[VARIABLE_DESCARTADA].isna().sum()} faltantes "
      f"({100*df_train[VARIABLE_DESCARTADA].isna().mean():.1f}%).")

medianas = {}
for var in PREDICTORES:
    if df_modelo[var].isna().any():
        mediana = float(df_modelo[var].median())
        medianas[var] = mediana
        df_modelo[var] = df_modelo[var].fillna(mediana)
        print(f"[IMPUTADA]  '{var}': mediana = {mediana:.2f}")

with open(DATA_DIR / "medianas_train.json", "w", encoding="utf-8") as f:
    json.dump(medianas, f, indent=2)
print(f"\n[GUARDADO] data/medianas_train.json")

n_final = len(df_modelo)
assert df_modelo[PREDICTORES + [VAR_RESPUESTA]].isna().sum().sum() == 0

print(f"\nFilas disponibles para el modelo: {n_final} de {n_total}")

# =====================================================================
# 6. DEFINICION DE X E Y
# =====================================================================
y     = df_modelo[VAR_RESPUESTA].values
X_raw = df_modelo[PREDICTORES].values
X     = sm.add_constant(X_raw)

# =====================================================================
# 7. AJUSTE
# =====================================================================
print("\n" + "=" * 62)
print(" 3. AJUSTE DEL MODELO LOGISTICO COMPLETO")
print("=" * 62)

modelo    = sm.Logit(y, X)
resultado = modelo.fit(disp=False)

print(f"[OK] Convergencia: {resultado.mle_retvals['converged']}")
print(f"Observaciones usadas : {int(resultado.nobs)}")
print(f"Parametros estimados : {len(resultado.params)}")

# =====================================================================
# 8. EXTRACCION DE RESULTADOS (CORRECCION: convertir a arrays numpy)
# =====================================================================
# params, bse, zvalues, pvalues pueden ser Series o arrays.
# Convertimos a arrays numpy para consistencia en operaciones.
coeficientes = np.asarray(resultado.params)       # shape: (8,)
errores_std  = np.asarray(resultado.bse)          # shape: (8,)
valores_z    = np.asarray(resultado.tvalues)      # shape: (8,)
valores_p    = np.asarray(resultado.pvalues)      # shape: (8,)

# Crear Series de pandas para acceso por NOMBRE (crucial en interpretacion)
index_labels = ["const"] + PREDICTORES
coef_serie   = pd.Series(coeficientes, index=index_labels)
se_serie     = pd.Series(errores_std, index=index_labels)
p_serie      = pd.Series(valores_p, index=index_labels)
or_serie     = pd.Series(np.exp(coeficientes), index=index_labels)

# =====================================================================
# 9. ODDS RATIOS E IC
# =====================================================================
odds_ratios = np.exp(coeficientes)                # array numpy
z_critico   = stats.norm.ppf(1 - SIGNIFICANCIA / 2)

ic_beta_inf = coeficientes - z_critico * errores_std
ic_beta_sup = coeficientes + z_critico * errores_std
ic_or_inf   = np.exp(ic_beta_inf)
ic_or_sup   = np.exp(ic_beta_sup)

# =====================================================================
# 10. TABLA RESUMEN (CORRECCION: sin .values)
# =====================================================================
tabla = pd.DataFrame({
    "Variable":      ["const"] + PREDICTORES,
    "Beta":          coeficientes,               # ya es numpy array
    "Error_Std":     errores_std,
    "Z":             valores_z,
    "P_valor":       valores_p,
    "Odds_Ratio":    odds_ratios,
    "IC95_inf":      ic_or_inf,
    "IC95_sup":      ic_or_sup,
})

pd.set_option("display.precision", 4)
print("\n" + "=" * 62)
print(" 4. RESULTADOS DEL MODELO COMPLETO")
print("=" * 62)
print(tabla.to_string(index=False))

with open(OUT_DIR / "etapa2_coeficientes.txt", "w", encoding="utf-8") as f:
    f.write("MODELO LOGISTICO COMPLETO - RESULTADOS\n")
    f.write("=" * 56 + "\n\n")
    f.write(f"Observaciones: {int(resultado.nobs)}\n")
    f.write(f"Predictores  : {', '.join(PREDICTORES)}\n")
    f.write(f"Variable descartada: {VARIABLE_DESCARTADA} (50% faltantes)\n\n")
    f.write(tabla.to_string(index=False))
    f.write("\n\nNotas:\n")
    f.write("- Beta: coeficiente en escala logit\n")
    f.write("- Odds_Ratio = exp(Beta)\n")
    f.write("- IC95 = exp(Beta +/- 1.96 * Error_Std)\n")

print("\n[GUARDADO] outputs/etapa2_coeficientes.txt")

# =====================================================================
# 11. INTERPRETACION (CORRECCION: usar Series para acceso por NOMBRE)
# =====================================================================
print("\n" + "=" * 62)
print(" 5. INTERPRETACION DE COEFICIENTES")
print("=" * 62)

VAR_INTERPRETAR = ["glu", "bmi", "age", "ped"]
NOMBRES = {
    "glu": "glucosa plasmatica (mg/dL)",
    "bmi": "indice de masa corporal (kg/m^2)",
    "age": "edad (anos)",
    "ped": "funcion pedigree (antecedentes familiares)",
}

lineas_informe = []

for var in VAR_INTERPRETAR:
    # ACCESO POR ETIQUETA usando las Series creadas arriba
    beta   = float(coef_serie[var])
    se     = float(se_serie[var])
    p_val  = float(p_serie[var])
    or_val = float(or_serie[var])
    ic_lo  = float(ic_or_inf[list(index_labels).index(var)])
    ic_hi  = float(ic_or_sup[list(index_labels).index(var)])

    es_sig = p_val < SIGNIFICANCIA
    etiq   = "SIGNIFICATIVO" if es_sig else "no significativo"

    bloque = []
    bloque.append(f"\n{var.upper()} -- {NOMBRES[var]}")
    bloque.append(f"  beta = {beta:.4f} | SE = {se:.4f} | p = {p_val:.4f} [{etiq}]")
    bloque.append(f"  Odds Ratio = {or_val:.3f} | IC 95% = [{ic_lo:.3f}, {ic_hi:.3f}]")

    if beta > 0:
        cambio = (or_val - 1) * 100
        bloque.append(
            f"  Por cada unidad adicional de {NOMBRES[var]}, el odd de "
            f"presentar diabetes se multiplica por {or_val:.3f} "
            f"(aumento del {cambio:.1f}%), manteniendo constantes las "
            f"demas variables del modelo."
        )
    else:
        cambio = (1 - or_val) * 100
        bloque.append(
            f"  Por cada unidad adicional de {NOMBRES[var]}, el odd de "
            f"presentar diabetes se multiplica por {or_val:.3f} "
            f"(reduccion del {cambio:.1f}%), manteniendo constantes las "
            f"demas variables del modelo."
        )

    if ic_lo <= 1 <= ic_hi:
        bloque.append("  El IC 95% incluye el valor 1: no hay evidencia concluyente.")
    else:
        bloque.append("  El IC 95% NO incluye el valor 1: evidencia consistente.")

    texto = "\n".join(bloque)
    print(texto)
    lineas_informe.append(texto)

# =====================================================================
# 12. RESUMEN GLOBAL
# =====================================================================
print("\n" + "=" * 62)
print(" 6. RESUMEN GLOBAL")
print("=" * 62)
print(f"Pseudo-R2 (McFadden) : {resultado.prsquared:.4f}")
print(f"Log-verosimilitud    : {resultado.llf:.4f}")
print(f"Log-verosimilitud nula: {resultado.llnull:.4f}")
print(f"AIC                  : {resultado.aic:.2f}")
print(f"BIC                  : {resultado.bic:.2f}")

with open(OUT_DIR / "etapa2_interpretaciones.txt", "w", encoding="utf-8") as f:
    f.write("INTERPRETACION DE COEFICIENTES - MODELO COMPLETO\n")
    f.write("=" * 56 + "\n")
    f.write(f"Observaciones: {int(resultado.nobs)} | "
            f"Pseudo-R2: {resultado.prsquared:.4f} | "
            f"AIC: {resultado.aic:.2f}\n")
    f.write("\n".join(lineas_informe))
    f.write("\n\nTodas las interpretaciones asumen que las demas "
            "variables permanecen constantes.\n")

print("\n[GUARDADO] outputs/etapa2_interpretaciones.txt")

print("\n" + "=" * 62)
print(" ETAPA 2 COMPLETADA")
print("=" * 62)
print("PROXIMO PASO: Etapa 3 - significancia global (modelo vs intercepto)")