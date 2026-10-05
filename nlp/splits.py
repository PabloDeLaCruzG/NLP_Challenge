"""Particiones del 10-CV: estratificadas, fijas y guardadas en disco (una por actividad)

Cada fila guarda el numero de pliegue (0..9) en el que actua como test,
el entrenamiento de ese pliegue son todas las demas. 
Asi cada fila es test exactamente una vez, por construccion.
Todos los modelos usan estas mismas particiones.
"""
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from data import ROOT

SEED = 42
N_SPLITS = 10
SPLITS_DIR = ROOT / "results" / "splits"


def fingerprint(data: pd.DataFrame) -> str:
    """Huella de las filas para detectar que el dataset ha cambiado"""
    keys = data[["year", "evaluator", "work", "section"]].astype(str).agg("-".join, axis=1)
    return hashlib.sha256("|".join(keys).encode()).hexdigest()[:16]


def assign_folds(data: pd.DataFrame, seed: int = SEED, n_splits: int = N_SPLITS) -> np.ndarray:
    """Para cada fila, el pliegue en el que es test. Estratifica por 'stratum' y baraja con la semilla"""
    fold_id = np.empty(len(data), dtype=int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for k, (_, test_idx) in enumerate(skf.split(data, data["stratum"])):
        fold_id[test_idx] = k
    return fold_id


def fold_indices(fold_id: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Posiciones de train y de test del pliegue k"""
    return np.where(fold_id != k)[0], np.where(fold_id == k)[0]


def get_or_create_splits(activity: int, data: pd.DataFrame) -> np.ndarray:
    """Carga las particiones guardadas, si no existen las crea, Falla si el dataset ya no es el mismo"""
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    path = SPLITS_DIR / f"activity{activity}_splits.json"
    current = fingerprint(data)
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["fingerprint"] != current:
            raise ValueError(f"{path.name}: el dataset ha cambiado, borra el fichero para regenerar las particiones")
        return np.array(saved["fold_id"])
    fold_id = assign_folds(data)
    payload = {"activity": activity, "seed": SEED, "n_splits": N_SPLITS, "n_rows": len(data),
               "fingerprint": current, "fold_id": fold_id.tolist()}
    path.write_text(json.dumps(payload))
    print(f"particiones creadas: {path.name}")
    return fold_id