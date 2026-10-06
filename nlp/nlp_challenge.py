"""
Desafio NLP: deteccion de incoherencias en evaluacion por pares (version unificada TEMPORAL).

Contenido actual (nivel basico, puntos de la memoria):
  1. Preprocesado de texto: minusculas, tokenizacion, eliminacion de stopwords, stemming.
  2. Validacion cruzada de 10 pliegues (estratificada, particiones fijas por semilla).
  3. Arnes de evaluacion (MAE y RMSE por pliegue, Wilcoxon) y baseline constante.
Pendiente de anadir: modelos (BoW + MLP y secuencial en PyTorch).

Dependencias: numpy, pandas, scipy, scikit-learn, nltk.
Un dataset y un modelo POR ACTIVIDAD (el notebook del profesor dice que las actividades son independientes).
"""
import re
import unicodedata
from pathlib import Path
from typing import Callable

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from scipy.stats import wilcoxon
from sklearn.model_selection import StratifiedKFold

# ----------------------------------------------------------------------------------------------
# Constantes
# ----------------------------------------------------------------------------------------------
CSV_SOURCE = "https://www.dlsi.ua.es/~juanra/UA/dataset/dcadep/dcadep_melt_grades.csv.gz"
RESULTS_DIR = Path("results")

ACTIVITIES = [1, 2]
COURSES = [2015, 2016]
SECTIONS = ['1', '2', '3', '4', '5', '6', '7'] 
TARGET = "value"

SEED = 42
N_SPLITS = 10

EMPTY_TOKEN = "<empty>" 

# Stopwords que se conservan
KEEP_STOPWORDS = {
    "no",
    "muy",
    "todo",
    "mas",
    "pero",
    "poco",
    "hay",
    "tiene",
    "nada",
    "ni",
    "si",
    "mucho",
    "sin",
    "todos",
    "tanto",
    "algo"
}


# ----------------------------------------------------------------------------------------------
# Datos
# ----------------------------------------------------------------------------------------------
def make_strata(values: pd.Series) -> pd.Series:
    """Estratos para balancear pliegues: <1, [1,2), [2,3) y 3 exacto"""
    conditions = [values < 1, values < 2, values < 3]
    return pd.Series(np.select(conditions, [0, 1, 2], default=3), index=values.index) # asigna 0, 1, 2, 3 depende la condicion, por defecto 3.


def load_activity(activity: int, source: str = CSV_SOURCE) -> pd.DataFrame:
    """Filas de una actividad con estratos y flag de feedback vacio"""
    df = pd.read_csv(source, sep="\t", decimal=",")
    df["feedback"] = df["feedback"].fillna("")
    df["section"] = df["section"].astype(str)  # mezcla de la 1 a la 7 con grade1 y grade2

    mask = (df["activity"] == activity) & df["section"].isin(SECTIONS) & df["year"].isin(COURSES)
    out = df.loc[mask, ["year", "evaluator", "work", "section", TARGET, "feedback"]].copy()

    # Aseguraciones de los datos
    assert out[TARGET].notna().all(), "hay NaN en value"
    assert out[TARGET].between(0, 3).all(), "value fuera de [0, 3]"

    out["is_empty"] = out["feedback"].str.strip() == ""
    out["stratum"] = make_strata(out[TARGET])
    return out.reset_index(drop=True)


# ----------------------------------------------------------------------------------------------
# Preprocesado de texto (SIN ESTADO: no aprende nada de los datos, asi que no hay fuga de datos)
# Lo que si aprende (vocabulario, IDF) se construira dentro de cada pliegue.
# ----------------------------------------------------------------------------------------------
def strip_accents(text: str) -> str:
    """Quita los acentos"""
    decomposed = unicodedata.normalize("NFD", text)  # separa cada letra de su marca de acento
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")  # Mn = marca: se descarta


def tokenize(text: str) -> list[str]:
    """Minusculas, sin acentos y solo letras o digitos"""
    return re.findall(r"[a-z0-9]+", strip_accents(text.lower()))


def build_stopwords(keep: set[str] = KEEP_STOPWORDS) -> set[str]:
    """Lista de NLTK normalizada igual que el texto (sin acentos) y sin las palabras a conservar"""
    nltk.download("stopwords", quiet=True)
    normalized = {strip_accents(w) for w in stopwords.words("spanish")}
    return normalized - {strip_accents(w) for w in keep}  # la lista a conservar se normaliza


def preprocess_text(text: str, stop_set: set[str], stemmer: SnowballStemmer) -> list[str]:
    """minusculas -> sin acentos -> tokens -> sin stopwords -> stemming, si no queda nada = <empty>"""
    tokens = [t for t in tokenize(text) if t not in stop_set]
    stems = [stemmer.stem(t) for t in tokens]
    return stems if stems else [EMPTY_TOKEN]


def preprocess_corpus(texts: pd.Series, keep: set[str] = KEEP_STOPWORDS) -> pd.Series:
    """Aplica el pipeline a una serie de textos y devuelve una serie de listas de tokens"""
    stop_set = build_stopwords(keep)
    stemmer = SnowballStemmer("spanish")
    return texts.map(lambda text: preprocess_text(text, stop_set, stemmer))


# ----------------------------------------------------------------------------------------------
# Validacion cruzada de 10 pliegues 
# Estratificada, misma semilla para todos los modelos
# ----------------------------------------------------------------------------------------------
def assign_folds(data: pd.DataFrame, seed: int = SEED, n_splits: int = N_SPLITS) -> np.ndarray:
    """
    Para cada fila, el pliegue (0..9) en el que es test. Estratifica por 'stratum' y baraja con la semilla.

    Cada fila es test exactamente una vez. Mismos pliegues para todos los modelos, para Wilcoxon
    """
    fold_id = np.empty(len(data), dtype=int) # crea array nuevo sin inicializar valores
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for k, (_, test_idx) in enumerate(skf.split(data, data["stratum"])):
        fold_id[test_idx] = k
    return fold_id


def fold_indices(fold_id: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Posiciones de train y de test del pliegue k"""
    train_idx = np.where(fold_id != k)[0]
    test_idx = np.where(fold_id == k)[0] # [0] para que devuelva directamente el array interno
    return train_idx, test_idx


# ----------------------------------------------------------------------------------------------
# Arnes de evaluacion
# Un modelo es cualquier objeto con fit(train_df) y predict(test_df) 
# Se crea uno NUEVO en cada pliegue
# ----------------------------------------------------------------------------------------------
class ConstantBaseline:
    """Predice siempre la misma constante, calculada con el entrenamiento (media o mediana)"""

    def __init__(self, statistic: str = "median"):
        assert statistic in ("mean", "median"), "statistic debe ser 'mean' o 'median'"
        self.statistic = statistic
        self.constant = None

    def fit(self, train_df: pd.DataFrame) -> None:
        y = train_df[TARGET].to_numpy()
        self.constant = float(np.mean(y) if self.statistic == "mean" else np.median(y))

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        return np.full(len(test_df), self.constant) # se copia el valor de constant en cada osicion del array


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """MAE y RMSE de unas predicciones"""
    error = y_pred - y_true
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    return {"mae": mae, "rmse": rmse}


def evaluate_cv(model_name: str, model_factory: Callable, data: pd.DataFrame, fold_id: np.ndarray) -> pd.DataFrame:
    """Entrena y evalua un modelo en cada pliegue. Devuelve una fila por pliegue con MAE, RMSE y MAE por curso"""
    rows = []
    for k in range(int(fold_id.max()) + 1):
        train_idx, test_idx = fold_indices(fold_id, k)
        train_df, test_df = data.iloc[train_idx], data.iloc[test_idx]
        model = model_factory()  # modelo nuevo en cada pliegue, asi nada se arrastra del anterior
        model.fit(train_df)
        y_true = test_df[TARGET].to_numpy()
        y_pred = model.predict(test_df)
        row = {"model": model_name, "fold": k, "n_test": len(test_df), **compute_metrics(y_true, y_pred)} # ** desempaquetan el diccionario que devuelve, se evita que se aniden diccionarios
        for year in sorted(data["year"].unique()):  # error por curso: comprueba que mezclar cursos no sesga
            mask = (test_df["year"] == year).to_numpy()
            row[f"mae_{year}"] = compute_metrics(y_true[mask], y_pred[mask])["mae"]
        rows.append(row)
    return pd.DataFrame(rows)


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
    _, p_value = wilcoxon(a, b)  # dos colas, con 10 pares el p minimo posible es 2/2^10 = 0.00195
    return {
        "metric": metric, 
        "a": model_a, 
        "b": model_b, 
        "median_diff_a_minus_b": float(np.median(a - b)),    
        "folds_a_better": int(np.sum(a < b)), 
        "p_value": float(p_value)
        }


# ----------------------------------------------------------------------------------------------
# Ejecucion
# ----------------------------------------------------------------------------------------------
def run_activity(activity: int, source: str = CSV_SOURCE) -> pd.DataFrame:
    """Carga, preprocesa, parte en pliegues y evalua los baselines de una actividad"""
    data = load_activity(activity, source)
    data["tokens"] = preprocess_corpus(data["feedback"])
    fold_id = assign_folds(data)

    vocab_size = len({token for tokens in data["tokens"] for token in tokens})
    print(f"\n=== Actividad {activity}: {len(data):,} filas | vocabulario tras el pipeline: {vocab_size:,} ===")
    print("filas de test por pliegue:", np.bincount(fold_id).tolist())

    results = pd.concat([
        evaluate_cv("const_mean", lambda: ConstantBaseline("mean"), data, fold_id),
        evaluate_cv("const_median", lambda: ConstantBaseline("median"), data, fold_id),
    ], ignore_index=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_DIR / f"activity{activity}_cv_results.csv", index=False)
    print(summarize_results(results).to_string())
    for metric in ("mae", "rmse"):
        print(" ", wilcoxon_compare(results, "const_mean", "const_median", metric))
    return results


def main() -> None:
    for activity in ACTIVITIES:
        run_activity(activity)


if __name__ == "__main__":
    main()