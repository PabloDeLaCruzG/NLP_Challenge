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

import re
import unicodedata

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

def make_strata(values: pd.Series) -> pd.Series:
    """Estratos para balancear pliegues: <1, [1,2), [2,3) y 3 exacto (nota perfecta)."""
    conditions = [values < 1, values < 2, values < 3]  # np.select toma la primera condicion cierta
    return pd.Series(np.select(conditions, [0, 1, 2], default=3), index=values.index)

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

    out["stratum"] = make_strata(out["value"])

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
    print("estratos:", data["stratum"].value_counts().sort_index().to_dict(),
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

def exact_values(activity: int, data: pd.DataFrame) -> None:
    """Valores exactos de value y reparto por curso de los que caen en .5"""
    counts = data["value"].round(2).value_counts().sort_index()  # round(2) evita ruido de coma flotante
    print(f"\nActividad {activity}: valores exactos de value")
    print(counts.to_string())
    is_half = pd.Series(np.isclose(data["value"] % 1, 0.5), index=data.index)  # %1 deja la parte decimal
    print("\nproporcion de filas con valor en .5, por curso:")
    print(is_half.groupby(data["year"]).mean().round(3).to_string())  # media de booleanos = proporcion
    print("\nvalores en .5 por curso:")
    print(data.loc[is_half].groupby("year")["value"].value_counts().sort_index().to_string())

def normalize_text(text: str) -> str:
    """Minusculas, sin acentos ni signos, espacios colapsados (solo para este analisis exploratorio)."""
    text = unicodedata.normalize("NFD", text.lower())  # separa cada letra de su acento
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")  # Mn = marca de acento: se descarta
    text = re.sub(r"[^a-z0-9 ]+", " ", text)  # fuera todo lo que no sea letra, digito o espacio
    return re.sub(r"\s+", " ", text).strip()

def filler_coverage(activity: int, data: pd.DataFrame, year: int, top_k=(5, 10, 25, 50)) -> None:
    """Que parte de un curso cubren los textos normalizados mas frecuentes y que nota llevan."""
    subset = data[data["year"] == year].copy()
    subset["norm"] = subset["feedback"].map(normalize_text)
    counts = subset["norm"].value_counts()
    print(f"\n--- Actividad {activity} | {year}: {len(counts)} textos normalizados distintos en {len(subset)} filas ---")
    for k in top_k:
        print(f"  los {k:2d} textos mas frecuentes cubren {counts.iloc[:k].sum() / len(subset):.1%} de las filas")
    n_words = subset["norm"].str.split().str.len()  # un texto vacio da 0 palabras
    short = n_words <= 3
    print(f"  textos de 3 palabras o menos: {short.mean():.1%} de las filas | nota media "
          f"{subset.loc[short, 'value'].mean():.2f} frente a {subset.loc[~short, 'value'].mean():.2f} en el resto")
    stats = subset.groupby("norm")["value"].agg(["size", "mean"]).sort_values("size", ascending=False)
    print(stats.head(12).round(2))

def main() -> None:
    pd.set_option("display.max_colwidth", 70)  # que no recorte los textos largos
    df = load_raw()
    print("filas totales del CSV:", len(df))
    datasets = {activity: load_activity(df, activity) for activity in ACTIVITIES}
    for activity, data in datasets.items():
        summarize_activity(activity, data)
        exact_values(activity, data)
    for activity, data in datasets.items():
        empty_vs_value(activity, data)
        top_texts(activity, data)
    for activity, data in datasets.items():
        for year in COURSES:
            filler_coverage(activity, data, year)
 
 
if __name__ == "__main__":
    main()
