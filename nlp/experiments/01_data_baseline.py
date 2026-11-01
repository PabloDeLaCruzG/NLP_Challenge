# Carga, filtrado y distribucion de 'value'

from pathlib import Path
import hashlib
import urllib.request

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    ROOT = Path(__file__).resolve().parents[1]   # el .py vive en desafios/nlp/
except NameError:                                # celda ejecutada en un kernel interactivo
    ROOT = Path.cwd()
assert (ROOT / "data").is_dir(), f"Ejecuta desde la raíz 'desafios' (ROOT={ROOT})"

RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True) # la crea si no existe
CSV_PATH = RAW_DIR / "dcadep_melt_grades.csv"
 
sha = hashlib.sha256(CSV_PATH.read_bytes()).hexdigest() # seguridad del csv
print(f"{CSV_PATH.name}: {CSV_PATH.stat().st_size:,} bytes · sha256={sha[:16]}…") # imprime el nombre, tamaño, y los primeros 16 bytes

# Carga cruda 
raw = pd.read_csv(CSV_PATH, sep="\t", decimal=",")
print("sahpe:", raw.shape)
print(raw.dtypes, "\n")
print("NaN por columna:\n", raw.isna().sum(), "\n")
print("valores de section:", sorted(raw["section"].astype(str).unique()))

# Filtrado, secciones 1-7
df = raw.copy()
df["feedback"] = df["feedback"].fillna("") # convierte null to ""
df["section"] = df["section"].astype(str) # converite a texto
mask = (df["activity"] == 1) & df["section"].isin(['1','2','3','4','5','6','7']) # condicional que sean actividad 1 y pertenezcan alguna seccion del 1 al 7

# df.loc[filas, columnas], mask = solo las filas donde es True. solo las columnas esas
a1 = df.loc[mask, ["section", "value", "feedback"]].reset_index(drop=True)

# Comprobaciones de calidad del dato, si devuelve False, se detiene el programa (AssertionError)
assert a1["value"].notna().all(), "hay NaN en value"
assert a1["value"].between(0,3).all(), "value fuera de [0,3]"
print("filas act1, secciones 1-7:", len(a1))

# Distribucion de value
v = a1["value"]
print(v.describe(), "\n")
print(f"value == 3   : {(v == 3).mean():.3f}")
print(f"value >= 2.5 : {(v >= 2.5).mean():.3f}")
print(f"value <  1   : {(v < 1).mean():.3f}")
print(f"feedback vacío: {(a1['feedback'].str.strip() == '').mean():.3f}")
 
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
bins = np.arange(0, 3.05, 0.1)
ax[0].hist(v, bins=bins, edgecolor="black"); ax[0].set_title("value (escala lineal)")
ax[1].hist(v, bins=bins, edgecolor="black"); ax[1].set_yscale("log"); ax[1].set_title("value (escala log)")
for a in ax: a.set_xlabel("value")
plt.tight_layout(); plt.show()

# Baselines constantes (EXPLORATORIO, in-sample; el oficial irá por validación cruzada)
def mae(y, c):  
    return float(np.mean(np.abs(y - c)))
def rmse(y, c): 
    return float(np.sqrt(np.mean((y - c) ** 2)))
 
y = v.to_numpy()
for nombre, c in [("media", y.mean()), ("mediana", np.median(y)), ("3.0", 3.0)]:
    print(f"{nombre:8s} c={c:.3f}  MAE={mae(y, c):.3f}  RMSE={rmse(y, c):.3f}")


# ¿Cuatas filas hay de verdad?
act1 = df[mask].copy()   # mismas filas que a1, pero conservando year/group/evaluator/work
print(act1.groupby(["year", "group"]).size(), "\n")
print(act1[["evaluator", "work"]].nunique(), "\n")
CLAVE = ["evaluator", "work", "section"]
print("filas duplicadas en", CLAVE, ":", int(act1.duplicated(CLAVE).sum()))
no_vacio = act1["feedback"].str.strip() != ""
print("feedback no vacío:", int(no_vacio.sum()), "| textos distintos:", act1.loc[no_vacio, "feedback"].nunique())

# Feedback vacío frente a value
vacio = a1["feedback"].str.strip() == ""
clase = a1["value"].round().astype(int)       # {0,1,2,3}
print(pd.crosstab(vacio, clase, normalize="index", rownames=["vacío"], colnames=["value≈"]).round(3), "\n")
print(f"media de value | vacío: {a1.loc[vacio, 'value'].mean():.3f} | no vacío: {a1.loc[~vacio, 'value'].mean():.3f}")

# Clases candidatas para estratificar
assert not np.isclose(a1["value"] % 1, 0.5).any(), "hay valores en .5: np.round los redondea al par"
print(clase.value_counts().sort_index())

# 10. ¿Se parecen los dos cursos?
print(act1.groupby("year")["value"].agg(["mean", "std"]).round(3))
print((act1["feedback"].str.strip() == "").groupby(act1["year"]).mean().round(3))


print(df[(df["activity"] == 2) & df["section"].isin(['1','2','3','4','5','6','7'])].groupby("year").size())

