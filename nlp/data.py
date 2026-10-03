"""
Carga y utilidades de datos del desafio NLP (modulo compartido por los demas scripts).
 
Un dataset por actividad (cursos 2015 y 2016 juntos), como indica el notebook del profesor.
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
