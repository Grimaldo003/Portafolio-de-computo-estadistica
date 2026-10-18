"""
=====================================================================
ETAPA 8 - ARBOL DE DECISION Y COMPARACION CON REGRESION LOGISTICA
(seccion 9 del documento de tarea)
Proyecto: Regresion Logistica - Clasificacion de Diabetes
---------------------------------------------------------------------
OBJETIVO: Ajustar un arbol de decision sobre el mismo conjunto de
entrenamiento, evaluarlo sobre el mismo conjunto de prueba y
compararlo con el modelo logistico clinico.
---------------------------------------------------------------------
INPUT : data/train.csv, data/test.csv, data/medianas_train.json
OUTPUT: outputs/etapa8_arbol_reglas.txt
        outputs/etapa8_arbol_grafico.png
        outputs/etapa8_comparacion.txt
        outputs/etapa8_roc_comparacion.png
        outputs/etapa8_importancias.png
---------------------------------------------------------------------
DECISIONES METODOLOGICAS
  1. La profundidad del arbol NO se fija a ojo: se selecciona por
     validacion cruzada estratificada (5 pliegues) maximizando AUC.
  2. El umbral de decision optimo de cada modelo se calcula sobre
     TRAIN (no sobre test), para no introducir optimismo artificial
     en la comparacion.
  3. Ambos modelos se evaluan sobre el MISMO test y con el MISMO
     procedimiento, de modo que la comparacion sea justa.
  4. class_weight='balanced' se trata como hiperparametro: infla las
     probabilidades y desplaza el significado del umbral 0.50.
=====================================================================
"""

# ---------------------------------------------------------------------
# 1. IMPORTACIONES
# ---------------------------------------------------------------------
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve

# ---------------------------------------------------------------------
# 2. CONFIGURACION
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR  = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

TRAIN_PATH    = DATA_DIR / "train.csv"
TEST_PATH     = DATA_DIR / "test.csv"
MEDIANAS_PATH = DATA_DIR / "medianas_train.json"

VAR_RESPUESTA       = "type"
VARIABLE_DESCARTADA = "insulin"
PRED_CLINICO        = ["npreg", "glu", "bmi", "ped"]   # modelo de referencia
SEMILLA             = 42
UMBRAL_FIJO         = 0.50

# ---------------------------------------------------------------------
# 3. CARGA Y PREPARACION
# ---------------------------------------------------------------------
print("=" * 68)
print(" 1. CARGA Y PREPARACION DE DATOS")
print("=" * 68)

df_train = pd.read_csv(TRAIN_PATH)
df_test  = pd.read_csv(TEST_PATH)

with open(MEDIANAS_PATH, encoding="utf-8") as f:
    medianas = json.load(f)

def preparar(df, medianas):
    """Elimina 'insulin' e imputa faltantes con las medianas de TRAIN."""
    df = df.drop(columns=[VARIABLE_DESCARTADA])
    for var, valor in medianas.items():
        df[var] = df[var].fillna(valor)
    return df

df_train_prep = preparar(df_train.copy(), medianas)
df_test_prep  = preparar(df_test.copy(),  medianas)

y_train = df_train_prep[VAR_RESPUESTA].values.astype(int)
y_test  = df_test_prep[VAR_RESPUESTA].values.astype(int)

print(f"Train : {len(y_train)} obs | positivos {int(y_train.sum())} "
      f"({100*y_train.mean():.1f}%)")
print(f"Test  : {len(y_test)} obs | positivos {int(y_test.sum())} "
      f"({100*y_test.mean():.1f}%)")
print(f"Predictores: {', '.join(PRED_CLINICO)}")

# ---------------------------------------------------------------------
# 4. FUNCIONES AUXILIARES
# ---------------------------------------------------------------------

def metricas(y_true, y_pred):
    """Calcula el bloque completo de metricas a partir de la matriz."""
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    def div(a, b):
        return a / b if b > 0 else float("nan")

    sens = div(tp, tp + fn)
    spec = div(tn, tn + fp)
    prec = div(tp, tp + fp)
    vpn  = div(tn, tn + fn)
    acc  = div(tp + tn, tp + fp + fn + tn)
    f1   = div(2 * prec * sens, prec + sens)

    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "Sensibilidad": sens,
        "Especificidad": spec,
        "Precision": prec,
        "VPN": vpn,
        "Exactitud": acc,
        "F1": f1,
        "Ex_Balanceada": div(sens + spec, 2),
        "Youden": sens + spec - 1,
    }

def umbral_youden(y_true, probs):
    """
    Devuelve el umbral que maximiza el indice de Youden.
    Se descartan los umbrales infinitos que devuelve roc_curve.
    """
    fpr, tpr, thr = roc_curve(y_true, probs)
    finitos = np.isfinite(thr)
    fpr, tpr, thr = fpr[finitos], tpr[finitos], thr[finitos]
    idx = np.argmax(tpr - fpr)
    return float(thr[idx]), float((tpr - fpr)[idx])

# ---------------------------------------------------------------------
# 5. MODELO LOGISTICO (REFERENCIA)
# ---------------------------------------------------------------------
print("\n" + "=" * 68)
print(" 2. MODELO LOGISTICO CLINICO")
print("=" * 68)

X_tr_log = sm.add_constant(df_train_prep[PRED_CLINICO])
X_te_log = sm.add_constant(df_test_prep[PRED_CLINICO])

res_log = sm.Logit(y_train, X_tr_log).fit(disp=False)

p_log_train = np.asarray(res_log.predict(X_tr_log))
p_log_test  = np.asarray(res_log.predict(X_te_log))

auc_log = roc_auc_score(y_test, p_log_test)
print(f"AIC (train) : {res_log.aic:.2f}")
print(f"AUC (test)  : {auc_log:.4f}")

# ---------------------------------------------------------------------
# 6. ARBOL DE DECISION CON SELECCION DE HIPERPARAMETROS POR CV
# ---------------------------------------------------------------------
print("\n" + "=" * 68)
print(" 3. ARBOL DE DECISION (busqueda por validacion cruzada)")
print("=" * 68)

X_tr_tree = df_train_prep[PRED_CLINICO].values
X_te_tree = df_test_prep[PRED_CLINICO].values

# Rejilla de hiperparametros: se limita la profundidad porque con 200
# observaciones un arbol profundo memoriza el ruido de entrenamiento.
rejilla = {
    "max_depth":        [1, 2, 3, 4, 5, 6, 8],
    "criterion":        ["gini", "entropy"],
    "class_weight":     [None, "balanced"],
    "min_samples_leaf": [1, 5, 10],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)

busqueda = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=SEMILLA),
    param_grid=rejilla,
    scoring="roc_auc",       # se optimiza AUC: no depende del umbral
    cv=cv,
    n_jobs=-1,
)
busqueda.fit(X_tr_tree, y_train)

arbol = busqueda.best_estimator_

print("Mejores hiperparametros encontrados:")
for clave, valor in busqueda.best_params_.items():
    print(f"  {clave:<18}: {valor}")
print(f"AUC medio en CV (train) : {busqueda.best_score_:.4f}")
print(f"Profundidad efectiva    : {arbol.get_depth()}")
print(f"Numero de hojas         : {arbol.get_n_leaves()}")

p_tree_train = arbol.predict_proba(X_tr_tree)[:, 1]
p_tree_test  = arbol.predict_proba(X_te_tree)[:, 1]

auc_tree = roc_auc_score(y_test, p_tree_test)
print(f"AUC (test)              : {auc_tree:.4f}")

# ---------------------------------------------------------------------
# 7. SELECCION DE UMBRAL SOBRE TRAIN (SIN FUGA DE INFORMACION)
# ---------------------------------------------------------------------
print("\n" + "=" * 68)
print(" 4. UMBRAL OPTIMO POR MODELO (calculado sobre TRAIN)")
print("=" * 68)

u_log,  y_log  = umbral_youden(y_train, p_log_train)
u_tree, y_tree = umbral_youden(y_train, p_tree_train)

print(f"Logistico : umbral = {u_log:.4f} (Youden train = {y_log:.4f})")
print(f"Arbol     : umbral = {u_tree:.4f} (Youden train = {y_tree:.4f})")
print("\nNota: estos umbrales se aplican tal cual al conjunto de prueba.")

# ---------------------------------------------------------------------
# 8. EVALUACION SOBRE TEST
# ---------------------------------------------------------------------
print("\n" + "=" * 68)
print(" 5. EVALUACION SOBRE EL CONJUNTO DE PRUEBA")
print("=" * 68)

escenarios = {
    "Logistico @0.50":        (p_log_test,  UMBRAL_FIJO),
    f"Logistico @{u_log:.2f}": (p_log_test,  u_log),
    "Arbol @0.50":            (p_tree_test, UMBRAL_FIJO),
    f"Arbol @{u_tree:.2f}":    (p_tree_test, u_tree),
}

filas = []
for nombre, (probs, umbral) in escenarios.items():
    pred = (probs >= umbral).astype(int)
    m = metricas(y_test, pred)
    m["AUC"] = roc_auc_score(y_test, probs)   # el AUC no depende del umbral
    fila = {"Escenario": nombre, "Umbral": umbral}
    fila.update({k: round(v, 4) for k, v in m.items()
                 if k not in ("TP", "FP", "FN", "TN")})
    fila["TP"] = m["TP"]; fila["FP"] = m["FP"]
    fila["FN"] = m["FN"]; fila["TN"] = m["TN"]
    filas.append(fila)

tabla = pd.DataFrame(filas)
cols_orden = ["Escenario", "Umbral", "TP", "FP", "FN", "TN", "Sensibilidad",
              "Especificidad", "Precision", "VPN", "Exactitud", "F1",
              "Ex_Balanceada", "Youden", "AUC"]
tabla = tabla[cols_orden]

print(tabla.to_string(index=False))

# ---------------------------------------------------------------------
# 9. GRAFICO: CURVAS ROC COMPARADAS
# ---------------------------------------------------------------------
print("\n" + "=" * 68)
print(" 6. GRAFICOS")
print("=" * 68)

fig, ax = plt.subplots(figsize=(8, 7))

fpr_l, tpr_l, _ = roc_curve(y_test, p_log_test)
fpr_t, tpr_t, _ = roc_curve(y_test, p_tree_test)

ax.plot(fpr_l, tpr_l, linewidth=2.2, color="crimson",
        label=f"Regresion logistica (AUC = {auc_log:.3f})")
ax.plot(fpr_t, tpr_t, linewidth=2.2, color="seagreen",
        label=f"Arbol de decision (AUC = {auc_tree:.3f})")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.5,
        label="Azar (AUC = 0.500)")

# Marcar el umbral 0.50 de cada modelo sobre su propia curva
for probs, color in ((p_log_test, "crimson"), (p_tree_test, "seagreen")):
    pred = (probs >= UMBRAL_FIJO).astype(int)
    m = metricas(y_test, pred)
    ax.scatter([1 - m["Especificidad"]], [m["Sensibilidad"]],
               s=140, color=color, edgecolor="black", zorder=4)

ax.set_xlabel("Tasa de falsos positivos (1 - Especificidad)")
ax.set_ylabel("Tasa de verdaderos positivos (Sensibilidad)")
ax.set_title("Comparacion de curvas ROC: logistica vs arbol",
             fontsize=13, fontweight="bold")
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "etapa8_roc_comparacion.png", dpi=150)
plt.show()

# --- Grafico del arbol ---
fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(arbol,
          feature_names=PRED_CLINICO,
          class_names=["Sin diabetes", "Con diabetes"],
          filled=True, rounded=True, fontsize=9, ax=ax)
ax.set_title("Arbol de decision ajustado (profundidad optima por CV)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_DIR / "etapa8_arbol_grafico.png", dpi=150)
plt.show()

# --- Importancia de variables ---
fig, ax = plt.subplots(figsize=(8, 4))
importancias = pd.Series(arbol.feature_importances_, index=PRED_CLINICO)
importancias = importancias.sort_values(ascending=True)
ax.barh(importancias.index, importancias.values, color="steelblue")
ax.set_xlabel("Importancia (reduccion de impureza ponderada)")
ax.set_title("Importancia de variables segun el arbol",
             fontsize=13, fontweight="bold")
for i, v in enumerate(importancias.values):
    ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(OUT_DIR / "etapa8_importancias.png", dpi=150)
plt.show()

# ---------------------------------------------------------------------
# 10. REGLAS EXTRAIDAS DEL ARBOL
# ---------------------------------------------------------------------
print("\n" + "=" * 68)
print(" 7. REGLAS DEL ARBOL")
print("=" * 68)

reglas = export_text(arbol, feature_names=PRED_CLINICO, decimals=3)
print(reglas)

# ---------------------------------------------------------------------
# 11. GUARDAR RESULTADOS
# ---------------------------------------------------------------------
with open(OUT_DIR / "etapa8_arbol_reglas.txt", "w", encoding="utf-8") as f:
    f.write("ARBOL DE DECISION - REGLAS Y ESTRUCTURA\n")
    f.write("=" * 60 + "\n\n")
    f.write("Hiperparametros seleccionados por validacion cruzada:\n")
    for clave, valor in busqueda.best_params_.items():
        f.write(f"  {clave:<18}: {valor}\n")
    f.write(f"  AUC medio en CV    : {busqueda.best_score_:.4f}\n")
    f.write(f"  Profundidad        : {arbol.get_depth()}\n")
    f.write(f"  Hojas              : {arbol.get_n_leaves()}\n\n")
    f.write("Importancia de variables:\n")
    for var, imp in arbol.feature_importances_.__class__ and \
            sorted(zip(PRED_CLINICO, arbol.feature_importances_),
                   key=lambda x: -x[1]):
        f.write(f"  {var:<8}: {imp:.4f}\n")
    f.write("\nReglas (formato texto):\n")
    f.write(reglas)

with open(OUT_DIR / "etapa8_comparacion.txt", "w", encoding="utf-8") as f:
    f.write("COMPARACION: REGRESION LOGISTICA vs ARBOL DE DECISION\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Umbral optimo logistico (train): {u_log:.4f}\n")
    f.write(f"Umbral optimo arbol (train)    : {u_tree:.4f}\n\n")
    f.write(tabla.to_string(index=False))
    f.write("\n\n")

    # Conclusion automatica
    if auc_log > auc_tree:
        dif = auc_log - auc_tree
        concl = (f"La regresion logistica supera al arbol en AUC por "
                 f"{dif:.4f}. El arbol no aporta ventaja discriminativa.")
    elif auc_tree > auc_log:
        dif = auc_tree - auc_log
        concl = (f"El arbol supera a la logistica en AUC por {dif:.4f}, "
                 f"aunque con probabilidades discretas y menor estabilidad.")
    else:
        concl = "Ambos modelos presentan un AUC practicamente identico."
    f.write("CONCLUSION\n" + "-" * 60 + "\n" + concl + "\n")

print("[GUARDADO] outputs/etapa8_arbol_reglas.txt")
print("[GUARDADO] outputs/etapa8_comparacion.txt")
print("[GUARDADO] outputs/etapa8_arbol_grafico.png")
print("[GUARDADO] outputs/etapa8_roc_comparacion.png")
print("[GUARDADO] outputs/etapa8_importancias.png")

print("\n" + "=" * 68)
print(" ETAPA 8 COMPLETADA")
print("=" * 68)