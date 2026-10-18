"""
=====================================================================
ETAPA 6 - METRICAS DE DESEMPENO Y CURVA ROC
(seccion 7 del documento de tarea)
Proyecto: Regresion Logistica - Clasificacion de Diabetes
---------------------------------------------------------------------
OBJETIVO: Calcular todas las metricas de clasificacion a partir de
la matriz de confusion (umbral 0.50) y construir la curva ROC con AUC.
---------------------------------------------------------------------
INPUT : outputs/etapa5_predicciones.csv
OUTPUT: outputs/etapa6_metricas.txt
        outputs/etapa6_roc.png
---------------------------------------------------------------------
NOTA SOBRE NUMPY 2.0+
    En versiones recientes de NumPy (>=2.0), np.trapz fue renombrado
    a np.trapezoid. Se usa np.trapezoid si existe, fallback a np.trapz.
    Para este calculo se usa directamente sklearn.metrics.roc_auc_score.
---------------------------------------------------------------------
FORMULAS (a partir de TP, FP, FN, TN)
    Exactitud            = (TP + TN) / N
    Sensibilidad (TPR)   = TP / (TP + FN)
    Especificidad        = TN / (TN + FP)
    Precision (PPV)      = TP / (TP + FP)
    VPN                  = TN / (TN + FN)
    F1                   = 2 * PPV * TPR / (PPV + TPR)
    Exactitud balanceada = (Sensibilidad + Especificidad) / 2
    AUC                  = area bajo la curva ROC (sklearn)
=====================================================================
"""

# ---------------------------------------------------------------------
# 1. IMPORTACIONES
# ---------------------------------------------------------------------
from pathlib import Path                  # rutas relativas
import numpy as np                        # operaciones numericas
import pandas as pd                       # tablas de datos
import matplotlib.pyplot as plt           # graficos
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)

# ---------------------------------------------------------------------
# 2. CONFIGURACION
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent   # Tarea1/
OUT_DIR  = BASE_DIR / "outputs"              # carpeta de salidas
OUT_DIR.mkdir(exist_ok=True)

PRED_PATH = OUT_DIR / "etapa5_predicciones.csv"

UMBRAL      = 0.50
MODEL_NAMES = ["Completo", "Reducido", "Clinico"]

# ---------------------------------------------------------------------
# 3. CARGA DE PREDICCIONES
# ---------------------------------------------------------------------
if not PRED_PATH.exists():
    raise FileNotFoundError(
        "No existe etapa5_predicciones.csv. Ejecuta primero 05_prediccion.py."
    )

df_pred = pd.read_csv(PRED_PATH)
y_real  = df_pred["y_real"].values.astype(int)

print("=" * 66)
print(" 1. CARGA DE PREDICCIONES")
print("=" * 66)
print(f"Observaciones : {len(y_real)}")
print(f"Positivos reales : {int(y_real.sum())} "
      f"(prevalencia {100*y_real.mean():.1f}%)")

# ---------------------------------------------------------------------
# 4. FUNCIONES DE METRICAS
# ---------------------------------------------------------------------

def matriz_confusion(y_real, y_pred):
    """
    Construye la matriz de confusion explícita.
      TP: predijo 1 y era 1
      FP: predijo 1 y era 0
      FN: predijo 0 y era 1
      TN: predijo 0 y era 0
    """
    tp = int(((y_pred == 1) & (y_real == 1)).sum())
    fp = int(((y_pred == 1) & (y_real == 0)).sum())
    fn = int(((y_pred == 0) & (y_real == 1)).sum())
    tn = int(((y_pred == 0) & (y_real == 0)).sum())
    return tp, fp, fn, tn

def calcular_metricas(tp, fp, fn, tn):
    """
    Calcula todas las metricas solicitadas por la tarea.
    Uso de funcion auxiliar para evitar division por 0.
    """
    def div(num, den):
        return num / den if den > 0 else float("nan")

    total = tp + fp + fn + tn
    sensibilidad = div(tp, tp + fn)          # TPR
    especificidad = div(tn, tn + fp)        # TNR
    precision    = div(tp, tp + fp)          # PPV
    vpn          = div(tn, tn + fn)          # NPV
    exactitud    = div(tp + tn, total)       # accuracy
    f1           = div(2 * precision * sensibilidad,
                       precision + sensibilidad)
    ex_bal       = div(sensibilidad + especificidad, 2)

    return {
        "Exactitud":     exactitud,
        "Sensibilidad":  sensibilidad,
        "Especificidad": especificidad,
        "Precision":     precision,
        "VPN":           vpn,
        "F1":            f1,
        "Ex_Balanceada": ex_bal,
    }

# ---------------------------------------------------------------------
# 5. CALCULO POR MODELO
# ---------------------------------------------------------------------
print("\n" + "=" * 66)
print(" 2. METRICAS POR MODELO (umbral 0.50)")
print("=" * 66)

resumen_modelos = {}

for nombre in MODEL_NAMES:
    clases = df_pred[f"clase_{nombre}"].values.astype(int)
    probs  = df_pred[f"p_{nombre}"].values

    # Matriz de confusion y metricas
    tp, fp, fn, tn = matriz_confusion(y_real, clases)
    metricas = calcular_metricas(tp, fp, fn, tn)

    # ROC y AUC usando sklearn (estándar de facto)
    auc = roc_auc_score(y_real, probs)
    fpr, tpr, _ = roc_curve(y_real, probs)

    resumen_modelos[nombre] = {
        "matriz": (tp, fp, fn, tn),
        "metricas": metricas,
        "fpr": fpr,
        "tpr": tpr,
        "auc": auc,
    }

    # --- Reporte por consola ---
    print(f"\n--- Modelo: {nombre} ---")
    print(f"Matriz de confusion:")
    print(f"            Predicho 0   Predicho 1")
    print(f"  Real 0   {tn:>8d}   {fp:>8d}")
    print(f"  Real 1   {fn:>8d}   {tp:>8d}")
    print(f"Metricas (umbral {UMBRAL}):")
    for metrica, valor in metricas.items():
        print(f"  {metrica:<15}: {valor:.4f}")
    print(f"  {'AUC':<15}: {auc:.4f}")

# ---------------------------------------------------------------------
# 6. TABLA COMPARATIVA
# ---------------------------------------------------------------------
print("\n" + "=" * 66)
print(" 3. TABLA COMPARATIVA")
print("=" * 66)

filas = []
for nombre in MODEL_NAMES:
    d = resumen_modelos[nombre]
    fila = {"Modelo": nombre}
    fila.update({k: round(v, 4) for k, v in d["metricas"].items()})
    fila["AUC"] = round(d["auc"], 4)
    filas.append(fila)

tabla = pd.DataFrame(filas)
print(tabla.to_string(index=False))

# ---------------------------------------------------------------------
# 7. GRAFICO: CURVAS ROC SUPERPUESTAS
# ---------------------------------------------------------------------
print("\n" + "=" * 66)
print(" 4. CURVA ROC")
print("=" * 66)

fig, ax = plt.subplots(figsize=(8, 7))

colores = {"Completo": "steelblue", "Reducido": "seagreen", "Clinico": "crimson"}
for nombre in MODEL_NAMES:
    d = resumen_modelos[nombre]
    ax.plot(d["fpr"], d["tpr"], linewidth=2, color=colores[nombre],
            label=f"{nombre} (AUC = {d['auc']:.3f})")

# Diagonal de referencia
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.5,
        label="Azar (AUC = 0.500)")

ax.set_xlabel("Tasa de falsos positivos (1 - Especificidad)")
ax.set_ylabel("Tasa de verdaderos positivos (Sensibilidad)")
ax.set_title("Curvas ROC de los tres modelos logisticos",
             fontsize=13, fontweight="bold")
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "etapa6_roc.png", dpi=150)
plt.show()

# ---------------------------------------------------------------------
# 8. GUARDAR RESULTADOS
# ---------------------------------------------------------------------
with open(OUT_DIR / "etapa6_metricas.txt", "w", encoding="utf-8") as f:
    f.write("METRICAS DE DESEMPENO - MODELOS LOGISTICOS\n")
    f.write(f"Umbral de clasificacion: {UMBRAL}\n")
    f.write("=" * 58 + "\n\n")

    for nombre in MODEL_NAMES:
        d = resumen_modelos[nombre]
        tp, fp, fn, tn = d["matriz"]
        f.write(f"[{nombre}]\n")
        f.write(f"  Matriz: TP={tp} FP={fp} FN={fn} TN={tn}\n")
        for metrica, valor in d["metricas"].items():
            f.write(f"  {metrica:<15}: {valor:.4f}\n")
        f.write(f"  {'AUC':<15}: {d['auc']:.4f}\n\n")

    f.write("TABLA COMPARATIVA\n")
    f.write("-" * 58 + "\n")
    f.write(tabla.to_string(index=False))
    f.write("\n")

print("\n[GUARDADO] outputs/etapa6_metricas.txt")
print("[GUARDADO] outputs/etapa6_roc.png")

print("\n" + "=" * 66)
print(" ETAPA 6 COMPLETADA")
print("=" * 66)
print("PROXIMO PASO: Etapa 7 - comparacion de umbrales (0.20 a 0.70)")