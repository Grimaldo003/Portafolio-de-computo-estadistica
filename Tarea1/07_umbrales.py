"""
=====================================================================
ETAPA 7 - COMPARACION DE UMBRALES (0.20 a 0.70)
(seccion 8 del documento de tarea)
Proyecto: Regresion Logistica - Clasificacion de Diabetes
---------------------------------------------------------------------
OBJETIVO: Evaluar el impacto de variar el umbral de clasificacion
sobre sensibilidad, especificidad y precision. Determinar cual
umbral es mas adecuado segun diferentes criterios clinicos.
---------------------------------------------------------------------
INPUT : outputs/etapa5_predicciones.csv
OUTPUT: outputs/etapa7_umbrales.csv
        outputs/etapa7_metricas_por_umbral.txt
        outputs/etapa7_curvas_umbral.png
---------------------------------------------------------------------
MODELO PRINCIPAL: Clinico (AUC mas alto, 4 variables, relevancia clinica)
---------------------------------------------------------------------
Criterios comunes para elegir umbral:
    - Maximo de Youden (sensibilidad + especificidad - 1)
    - Sensibilidad >= 0.80 (priorizar no perder casos)
    - Equilibrio (maximo de F1)
    - Coste clinico (dependiendo de la severidad relativa)
=====================================================================
"""

# ---------------------------------------------------------------------
# 1. IMPORTACIONES
# ---------------------------------------------------------------------
from pathlib import Path                  # rutas relativas
import numpy as np                        # operaciones numericas
import pandas as pd                       # tablas de datos
import matplotlib.pyplot as plt           # graficos
from sklearn.metrics import roc_curve     # para verificacion cruzada

# ---------------------------------------------------------------------
# 2. CONFIGURACION
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent   # Tarea1/
OUT_DIR  = BASE_DIR / "outputs"              # carpeta de salidas
OUT_DIR.mkdir(exist_ok=True)

PRED_PATH = OUT_DIR / "etapa5_predicciones.csv"

# Modelo principal para esta etapa
MODEL_NAME = "Clinico"

# Umbrales a evaluar (0.20 a 0.70 en pasos de 0.05)
UMBRALS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

# ---------------------------------------------------------------------
# 3. CARGA DE PREDICCIONES
# ---------------------------------------------------------------------
if not PRED_PATH.exists():
    raise FileNotFoundError(
        "No existe etapa5_predicciones.csv. Ejecuta primero 05_prediccion.py."
    )

df_pred = pd.read_csv(PRED_PATH)
y_real  = df_pred["y_real"].values.astype(int)
probs   = df_pred[f"p_{MODEL_NAME}"].values

print("=" * 66)
print(" 1. CARGA DE PREDICCIONES")
print("=" * 66)
print(f"Observaciones : {len(y_real)}")
print(f"Positivos reales : {int(y_real.sum())} "
      f"(prevalencia {100*y_real.mean():.1f}%)")
print(f"Modelo evaluado: {MODEL_NAME}")

# ---------------------------------------------------------------------
# 4. FUNCIONES AUXILIARES
# ---------------------------------------------------------------------

def calcular_metrics_umbral(y_real, probs, umbral):
    """
    Clasifica con un umbral dado y calcula todas las metricas.
    """
    y_pred = (probs >= umbral).astype(int)

    tp = int(((y_pred == 1) & (y_real == 1)).sum())
    fp = int(((y_pred == 1) & (y_real == 0)).sum())
    fn = int(((y_pred == 0) & (y_real == 1)).sum())
    tn = int(((y_pred == 0) & (y_real == 0)).sum())

    total = tp + fp + fn + tn

    def div(num, den):
        return num / den if den > 0 else float("nan")

    sens = div(tp, tp + fn)               # TPR
    spec = div(tn, tn + fp)              # TNR
    prec = div(tp, tp + fp)               # PPV
    vpn  = div(tn, tn + fn)               # NPV
    acc  = div(tp + tn, total)            # accuracy
    f1   = div(2 * prec * sens, prec + sens)
    youden = sens + spec - 1              # indice de Youden

    return {
        "umbral": umbral,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Sensibilidad": sens,
        "Especificidad": spec,
        "Precision": prec,
        "VPN": vpn,
        "Exactitud": acc,
        "F1": f1,
        "Youden": youden,
    }

# ---------------------------------------------------------------------
# 5. CALCULO PARA CADA UMBRAL
# ---------------------------------------------------------------------
print("\n" + "=" * 66)
print(" 2. EVALUACION POR UMBRAL")
print("=" * 66)

resultados = []
for u in UMBRALS:
    m = calcular_metrics_umbral(y_real, probs, u)
    resultados.append(m)

df_resultados = pd.DataFrame(resultados)

# Mostrar tabla en consola
print("\nResumen por umbral:")
cols_reporte = ["umbral", "Sensibilidad", "Especificidad", "Precision", 
                "F1", "Youden", "Exactitud"]
print(df_resultados[cols_reporte].to_string(index=False))

# ---------------------------------------------------------------------
# 6. SELECCION DE UMBRALES OPTIMOS SEGUN CRITERIOS
# ---------------------------------------------------------------------
print("\n" + "=" * 66)
print(" 3. UMBRALES OPTIMOS SEGUN CRITERIOS")
print("=" * 66)

# Criterio 1: Maximo indice de Youden (equilibrio TPR-TNR)
idx_youden = df_resultados["Youden"].idxmax()
mejor_youden = df_resultados.loc[idx_youden]
print(f"\nCriterio 1: Maximo Youden (TPR + TNR - 1)")
print(f"  Umbral = {mejor_youden['umbral']:.2f}")
print(f"  Sensibilidad = {mejor_youden['Sensibilidad']:.3f}")
print(f"  Especificidad = {mejor_youden['Especificidad']:.3f}")
print(f"  Youden = {mejor_youden['Youden']:.3f}")

# Criterio 2: Sensibilidad >= 0.80 (priorizar no perder casos)
df_sens80 = df_resultados[df_resultados["Sensibilidad"] >= 0.80]
if len(df_sens80) > 0:
    # Entre los que cumplen sens >= 0.80, tomar el de mayor espec
    idx_spec_max = df_sens80["Especificidad"].idxmax()
    mejor_sens80 = df_resultados.loc[idx_spec_max]
    print(f"\nCriterio 2: Sensibilidad >= 0.80")
    print(f"  Umbral = {mejor_sens80['umbral']:.2f}")
    print(f"  Sensibilidad = {mejor_sens80['Sensibilidad']:.3f}")
    print(f"  Especificidad = {mejor_sens80['Especificidad']:.3f}")
else:
    mejor_sens80 = None
    print(f"\nCriterio 2: Ningun umbral alcanza sensibilidad >= 0.80")

# Criterio 3: Maximo F1 (equilibrio precisin-sensibilidad)
idx_f1 = df_resultados["F1"].idxmax()
mejor_f1 = df_resultados.loc[idx_f1]
print(f"\nCriterio 3: Maximo F1")
print(f"  Umbral = {mejor_f1['umbral']:.2f}")
print(f"  Sensibilidad = {mejor_f1['Sensibilidad']:.3f}")
print(f"  Precision = {mejor_f1['Precision']:.3f}")
print(f"  F1 = {mejor_f1['F1']:.3f}")

# Criterio 4: Umbral actual (0.50) para referencia
df_umbral_05 = df_resultados[df_resultados["umbral"] == 0.50]
if len(df_umbral_05) > 0:
    umbral_actual = df_umbral_05.iloc[0]
    print(f"\nCriterio 4: Umbral actual (0.50)")
    print(f"  Sensibilidad = {umbral_actual['Sensibilidad']:.3f}")
    print(f"  Especificidad = {umbral_actual['Especificidad']:.3f}")
    print(f"  F1 = {umbral_actual['F1']:.3f}")

# ---------------------------------------------------------------------
# 7. GRAFICOS
# ---------------------------------------------------------------------
print("\n" + "=" * 66)
print(" 4. GRAFICOS")
print("=" * 66)

# Fig 1: Metricas vs umbral (lineas)
fig1, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(df_resultados["umbral"], df_resultados["Sensibilidad"],
         marker='o', linewidth=2, label='Sensibilidad', color='crimson')
ax1.plot(df_resultados["umbral"], df_resultados["Especificidad"],
         marker='s', linewidth=2, label='Especificidad', color='steelblue')
ax1.plot(df_resultados["umbral"], df_resultados["Precision"],
         marker='^', linewidth=2, label='Precision', color='seagreen')
ax1.axvline(0.50, color='gray', linestyle='--', linewidth=1.5,
            label='Umbral actual (0.50)')

ax1.set_xlabel("Umbral de clasificacion")
ax1.set_ylabel("Métrica")
ax1.set_title("Efecto del umbral sobre las principales métricas",
              fontsize=13, fontweight="bold")
ax1.legend(loc="lower left")
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig(OUT_DIR / "etapa7_metricas_vs_umbral.png", dpi=150)
plt.show()

# Fig 2: Youden y F1 vs umbral (barras)
fig2, ax2 = plt.subplots(figsize=(10, 5))

x_vals = df_resultados["umbral"].values
ax2.bar(x_vals, df_resultados["Youden"], alpha=0.7, 
        label='Índice de Youden', color='mediumpurple')
ax2.scatter([mejor_youden["umbral"]], [mejor_youden["Youden"]],
            s=150, color='darkorange', zorder=3, label='Máximo Youden')

ax2.axvline(0.50, color='gray', linestyle='--', linewidth=1.5,
            label='Umbral actual (0.50)')
ax2.set_xlabel("Umbral de clasificación")
ax2.set_ylabel("Índice de Youden")
ax2.set_title("Índice de Youden por umbral",
              fontsize=13, fontweight="bold")
ax2.legend(loc="upper left")
ax2.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(OUT_DIR / "etapa7_youden_vs_umbral.png", dpi=150)
plt.show()

# Fig 3: Curva TPR vs FPR (ROC alternativa)
fig3, ax3 = plt.subplots(figsize=(8, 7))

fpr_curve, tpr_curve, thresholds_roc = roc_curve(y_real, probs)
ax3.plot(fpr_curve, tpr_curve, linewidth=2, label=f'{MODEL_NAME}', 
         color='crimson')
ax3.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1.5,
         label='Azar (AUC = 0.500)')

# Marcar umbral 0.50 en la curva
idx_05 = np.argmin(np.abs(thresholds_roc - 0.50))
ax3.scatter([fpr_curve[idx_05]], [tpr_curve[idx_05]], s=150,
            color='darkorange', zorder=3, label='Umbral 0.50')

ax3.set_xlabel("Tasa de falsos positivos (1 - Especificidad)")
ax3.set_ylabel("Tasa de verdaderos positivos (Sensibilidad)")
ax3.set_title("Curva ROC con marca de umbral 0.50",
              fontsize=13, fontweight="bold")
ax3.legend(loc="lower right")
ax3.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "etapa7_roc_marked.png", dpi=150)
plt.show()

# ---------------------------------------------------------------------
# 8. GUARDAR RESULTADOS
# ---------------------------------------------------------------------
df_resultados.to_csv(OUT_DIR / "etapa7_umbrales.csv", index=False)

with open(OUT_DIR / "etapa7_metricas_por_umbral.txt", "w", encoding="utf-8") as f:
    f.write("METRICAS POR UMBRAL - MODELO CLINICO\n")
    f.write("=" * 58 + "\n\n")
    f.write("Resumen por umbral:\n")
    f.write("-" * 58 + "\n")
    f.write(df_resultados[cols_reporte].to_string(index=False))
    f.write("\n\n")

    f.write("SELECCION DE UMBRAL SEGUN CRITERIOS\n")
    f.write("=" * 58 + "\n\n")

    f.write(f"Criterio 1: Maximo Youden\n")
    f.write(f"  Umbral = {mejor_youden['umbral']:.2f}\n")
    f.write(f"  Sensibilidad = {mejor_youden['Sensibilidad']:.4f}\n")
    f.write(f"  Especificidad = {mejor_youden['Especificidad']:.4f}\n")
    f.write(f"  Youden = {mejor_youden['Youden']:.4f}\n\n")

    if mejor_sens80 is not None:
        f.write(f"Criterio 2: Sensibilidad >= 0.80\n")
        f.write(f"  Umbral = {mejor_sens80['umbral']:.2f}\n")
        f.write(f"  Sensibilidad = {mejor_sens80['Sensibilidad']:.4f}\n")
        f.write(f"  Especificidad = {mejor_sens80['Especificidad']:.4f}\n\n")
    else:
        f.write(f"Criterio 2: Ningun umbral alcanza sensibilidad >= 0.80\n\n")

    f.write(f"Criterio 3: Maximo F1\n")
    f.write(f"  Umbral = {mejor_f1['umbral']:.2f}\n")
    f.write(f"  Sensibilidad = {mejor_f1['Sensibilidad']:.4f}\n")
    f.write(f"  Precision = {mejor_f1['Precision']:.4f}\n")
    f.write(f"  F1 = {mejor_f1['F1']:.4f}\n\n")

    f.write(f"Criterio 4: Umbral actual (0.50)\n")
    f.write(f"  Sensibilidad = {umbral_actual['Sensibilidad']:.4f}\n")
    f.write(f"  Especificidad = {umbral_actual['Especificidad']:.4f}\n")
    f.write(f"  F1 = {umbral_actual['F1']:.4f}\n\n")

    f.write("RECOMENDACION CLINICA\n")
    f.write("-" * 58 + "\n")
    f.write("El umbral 0.50 deja de detectar aproximadamente un 35% de los ")
    f.write(f"casos con diabetes (sensibilidad = {umbral_actual['Sensibilidad']:.2f}).\n")
    f.write(f"Para alcanzar sensibilidad >= 0.80 se requiere bajar a umbral ")
    if mejor_sens80 is not None:
        f.write(f"{mejor_sens80['umbral']:.2f}, sacrificando especificidad a ")
        f.write(f"{mejor_sens80['Especificidad']:.2f}.\n")
    else:
        f.write("0.20, con especificidad de X.XX (fuera de rango).\n")

print(f"\n[GUARDADO] outputs/etapa7_umbrales.csv")
print("[GUARDADO] outputs/etapa7_metricas_por_umbral.txt")
print("[GUARDADO] outputs/etapa7_metricas_vs_umbral.png")
print("[GUARDADO] outputs/etapa7_youden_vs_umbral.png")
print("[GUARDADO] outputs/etapa7_roc_marked.png")

print("\n" + "=" * 66)
print(" ETAPA 7 COMPLETADA")
print("=" * 66)
print("PROXIMO PASO: Etapa 8 - arbol de decision y comparacion")