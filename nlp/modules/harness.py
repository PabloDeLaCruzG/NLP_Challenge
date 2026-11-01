"""
Arnes de evaluacion: 10-CV por pliegue, resultados en CSV y comparacion con Wilcoxon

Un modelo es cualquier objeto con fit(train_df) y predict(test_df). El arnes crea un modelo NUEVO en
cada pliegue (reinicializacion limpia) y solo le da el entrenamiento para aprender. Asi el baseline y los
modelos de verdad se evaluan exactamente con el mismo protocolo
"""
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from splits import fold_indices

TARGET = "value"


class ConstantBaseline:
    """Predice siempre la misma constante, calculada con el entrenamiento (media o mediana)."""

    def __init__(self, statistic: str = "median"):
        assert statistic in ("mean", "median"), "statistic debe ser 'mean' o 'median'"
        self.statistic = statistic
        self.constant = None

    def fit(self, train_df: pd.DataFrame) -> None:
        y = train_df[TARGET].to_numpy()
        self.constant = float(np.mean(y) if self.statistic == "mean" else np.median(y))

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        return np.full(len(test_df), self.constant)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """MAE y RMSE de unas predicciones"""
    error = y_pred - y_true
    return {"mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error ** 2)))}


def evaluate_cv(model_name: str, model_factory: Callable, data: pd.DataFrame, fold_id: np.ndarray) -> pd.DataFrame:
    """Entrena y evalua un modelo en cada pliegue. Devuelve una fila por pliegue con MAE, RMSE y MAE por curso"""
    rows = []
    for k in range(int(fold_id.max()) + 1):
        train_idx, test_idx = fold_indices(fold_id, k)
        train_df, test_df = data.iloc[train_idx], data.iloc[test_idx]
        model = model_factory()  # modelo nuevo en cada pliegue: nada se arrastra del anterior
        model.fit(train_df)
        y_true = test_df[TARGET].to_numpy()
        y_pred = model.predict(test_df)
        row = {"model": model_name, "fold": k, "n_test": len(test_df), **compute_metrics(y_true, y_pred)}
        for year in sorted(data["year"].unique()):  # error por curso: comprueba que mezclar cursos no sesga
            mask = (test_df["year"] == year).to_numpy()
            row[f"mae_{year}"] = compute_metrics(y_true[mask], y_pred[mask])["mae"]
        rows.append(row)
    return pd.DataFrame(rows)


def save_results(new_rows: pd.DataFrame, path: Path) -> None:
    """Guarda resultados en CSV: sustituye las filas del mismo modelo y conserva las de los demas"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_csv(path)
        old = old[~old["model"].isin(new_rows["model"].unique())]
        new_rows = pd.concat([old, new_rows], ignore_index=True)
    new_rows.to_csv(path, index=False)


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    """Media y desviacion tipica entre pliegues de MAE y RMSE, por modelo"""
    summary = results.groupby("model").agg(
        mae_mean=("mae", "mean"), mae_std=("mae", "std"),
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
    )
    return summary.round(4)


def wilcoxon_compare(results: pd.DataFrame, model_a: str, model_b: str, metric: str = "mae") -> dict:
    """Wilcoxon de rangos con signo sobre las diferencias pareadas por pliegue entre dos modelos"""
    a = results[results["model"] == model_a].sort_values("fold")[metric].to_numpy()
    b = results[results["model"] == model_b].sort_values("fold")[metric].to_numpy()
    assert len(a) == len(b), "los dos modelos deben tener los mismos pliegues"
    statistic, p_value = wilcoxon(a, b)  # dos colas; con 10 pares el p minimo posible es 2/2^10 = 0.00195
    return {"metric": metric, "a": model_a, "b": model_b, "median_diff_a_minus_b": float(np.median(a - b)),
            "folds_a_better": int(np.sum(a < b)), "p_value": float(p_value)}