"""
Datos de las dos actividades y los dos cursos.

Un dataset por actividad (cursos juntos), porque el notebook de ejemplo dice que las actividades
son indepes y lo logico es un modelo por actividad.

Este fichero hace el analisis exploratorio y calcula los baselines contantes
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]   # el .py vive en desafios/nlp/
RAW_DIR = ROOT / "data" / "raw"
CSV_PATH = RAW_DIR / "dcadep_melt_grades.csv"

ACTIVITIES = [1,2]
COURSES = [2015, 2016]
SECTIONS = ['1','2','3','4','5','6','7']


def load_raw() -> pd.DataFrame:
    """
    Carga el CSV completo y normaliza tipos
     - feedback sin NaN
     - section como texto
    """
    path = CSV_PATH
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    print(f"CSV: {path.name}, sha256={digest}")

    df = pd.read_csv(path, sep="\t", decimal=",")
    df["feedback"] = df["feedback"].fillna("")
    df["section"] = df["section"].astype(str)

    return df

def load_activity(df: pd.DataFrame, activity: int, courses=COURSES) -> pd.DataFrame:
    """
    Cargar las filas de una actividad con columnas auxiliares
    """
    # Condicional de si la actividad, la seccion y el curso son correctas
    mask = (
        (df["activity"] == activity)
        & df["section"].isin(SECTIONS)
        & df["year"].isin(courses)
    )

    out = df.loc[mask, ["year", "evaluator", "work", "section", "value", "feedback"]].copy()

    # Aseguradores de que los datos estan limpios
    assert out["value"].notna().all(), "hay NaN en value"
    assert out["value"].between(0, 3).all(), "value fuera de [0,3]"

    out["is_empty"] = out["feedback"].str.strip() == ""

    # np.round redondea los .5 al par (2.5 -> 2), asi que avisamos si existieran
    n_halves = int(np.isclose(out["value"] % 1, 0.5).sum())
    if n_halves:
        print(f"AVISO actividad {activity}: {n_halves} valores en .5, el redondeo los manda al par")
    out["stratum"] = out["value"].round().astype(int)  # clases {0,1,2,3} para estratificar mas adelante

    return out.reset_index(drop=True)

def mae(y: np.ndarray, c: float) -> float:
    """Error Absoluto Medio de predecir la constante c"""
    mae = np.mean(np.abs(y - c))
    return float(mae)

def rmse(y: np.ndarray, c: float) -> float:
    """Raiz del error cuadratico medio de predecir la constante c"""
    rmse = np.sqrt(np.mean((y - c) ** 2))
    return float(rmse)

def summarize_activity(activity: int, data: pd.DataFrame):
    """
    Tamaño, resumen por curso, clases y baselines constantes de una activiadad
    """
    y = data["value"].to_numpy()
    print(f"\nActividad {activity}: {len(data):,} filas")
    per_course = data.groupby("year").agg(
        rows=("value", "size"),
        mean_value=("value", "mean"),
        empty_rate=("is_empty", "mean"),  # media de booleanos = proporcion de True
    )
    print(per_course.round(3))
    print("clases round(value):", data["stratum"].value_counts().sort_index().to_dict(),
          f"| value==3: {(y == 3).mean():.3f}")
    for name, c in [("media", y.mean()), ("mediana", np.median(y))]:
        print(f"  constante {name:8s} c={c:.3f}  MAE={mae(y, c):.3f}  RMSE={rmse(y, c):.3f}")

def empty_vs_value(activity: int, data: pd.DataFrame, year: int = 2015) -> None:
    """Distribucion de notas con feedback vacio y no vacio dentro de un solo curso."""
    subset = data[data["year"] == year]
    table = pd.crosstab(subset["is_empty"], subset["stratum"], normalize="index",
                        rownames=["vacio"], colnames=["nota"])
    print(f"\n--- Actividad {activity} | solo {year}: vacio frente a nota (proporciones por fila) ---")
    print(table.round(3))
 
def top_texts(activity: int, data: pd.DataFrame, year: int = 2016, n: int = 8) -> None:
    """Textos mas frecuentes de un curso, con su numero de apariciones y nota media."""
    subset = data[data["year"] == year]
    stats = subset.groupby("feedback")["value"].agg(["size", "mean"])
    print(f"\n--- Actividad {activity} | {year}: textos mas frecuentes ---")
    print(stats.sort_values("size", ascending=False).head(n).round(2))

def main() -> None:
    pd.set_option("display.max_colwidth", 70)  # que no recorte los textos largos
    df = load_raw()
    print("filas totales del CSV:", len(df))
    datasets = {activity: load_activity(df, activity) for activity in ACTIVITIES}
    for activity, data in datasets.items():
        summarize_activity(activity, data)
    for activity, data in datasets.items():
        empty_vs_value(activity, data)
        top_texts(activity, data)
 
 
if __name__ == "__main__":
    main()
