"""Bloque 6: crea o carga las particiones 10-CV de cada actividad y comprueba que estan bien hechas"""
import numpy as np
import pandas as pd

from data import ACTIVITIES, load_activity, load_raw
from splits import assign_folds, fold_indices, get_or_create_splits


def check_splits(activity: int, data: pd.DataFrame, fold_id: np.ndarray) -> None:
    """Tamaños, reparto de estratos y de cursos por pliegue, y propiedades que deben cumplirse siempre"""
    n_splits = int(fold_id.max()) + 1
    print(f"\n Actividad {activity}: {len(data):,} filas, {n_splits} pliegues")
    print("filas de test por pliegue:", np.bincount(fold_id).tolist())

    print("\nfilas de cada estrato en el test de cada pliegue:")
    print(pd.crosstab(fold_id, data["stratum"], rownames=["pliegue"], colnames=["estrato"]).to_string())

    global_share = data["stratum"].value_counts(normalize=True).sort_index()
    fold_share = pd.crosstab(fold_id, data["stratum"], normalize="index")
    max_dev = (fold_share - global_share).abs().max() * 100  # puntos porcentuales
    print("\ndesviacion maxima respecto al reparto global (puntos %), por estrato:", max_dev.round(2).to_dict())

    year_share = pd.crosstab(fold_id, data["year"], normalize="index")
    print("proporcion de 2015 por pliegue:", year_share[2015].round(3).tolist(), f"| global: {(data['year'] == 2015).mean():.3f}")

    for k in range(n_splits):  # propiedades que no pueden fallar nunca
        train_idx, test_idx = fold_indices(fold_id, k)
        assert len(np.intersect1d(train_idx, test_idx)) == 0, "train y test se solapan"
        assert len(train_idx) + len(test_idx) == len(data), "faltan filas"
    assert np.array_equal(fold_id, assign_folds(data)), "mismas semilla y datos deberian dar los mismos pliegues"
    print("comprobaciones OK: sin solapes, cada fila es test una vez, reproducible con la semilla")


def main() -> None:
    df = load_raw()
    for activity in ACTIVITIES:
        data = load_activity(df, activity)
        check_splits(activity, data, get_or_create_splits(activity, data))


if __name__ == "__main__":
    main()
