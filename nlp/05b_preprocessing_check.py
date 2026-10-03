"""Bloque 5b: comprobacion del pipeline de preprocesado (ejemplos y estadisticas por actividad)"""
import pandas as pd
from nltk.stem.snowball import SnowballStemmer

from data import ACTIVITIES, load_activity, load_raw
from preprocessing import EMPTY_TOKEN, build_stopwords, preprocess_corpus, preprocess_text, tokenize

EXAMPLES = [
    "Todo correcto",
    "Nada que objetar.",
    "No se puede acceder a la presentaci\u00f3n",  # \u00f3 = o con acento
    "Falta la imagen y no est\u00e1 bien referenciada",
    "Est\u00e1 muy bien, pero faltar\u00eda m\u00e1s informaci\u00f3n",
    "ok",
    "",
    "...",
]


def show_examples() -> None:
    """Muestra el resultado del pipeline sobre frases concretas ya vistas en los datos"""
    stop_set = build_stopwords()
    stemmer = SnowballStemmer("spanish")
    print("--- ejemplos ---")
    for text in EXAMPLES:
        print(f"{text!r:55s} -> {preprocess_text(text, stop_set, stemmer)}")


def vocabulary_size(token_lists: pd.Series) -> int:
    """Numero de tokens distintos en toda la serie."""
    return len({token for tokens in token_lists for token in tokens})


def corpus_stats(activity: int, data: pd.DataFrame) -> None:
    """Vocabulario, longitudes y filas que quedan vacias, antes y despues del pipeline."""
    raw_tokens = data["feedback"].map(tokenize)  # solo minusculas, sin acentos y tokenizado
    prep = preprocess_corpus(data["feedback"])
    lengths = prep.map(len)
    only_empty = prep.map(lambda tokens: tokens == [EMPTY_TOKEN])
    print(f"\n=== Actividad {activity} ({len(data):,} filas) ===")
    print(f"vocabulario: {vocabulary_size(raw_tokens):,} tras tokenizar -> {vocabulary_size(prep):,} tras el pipeline completo")
    print(f"longitud media: {raw_tokens.map(len).mean():.1f} -> {lengths.mean():.1f} tokens")
    print(f"longitud percentiles 50/90/99/max: {lengths.quantile([0.5, 0.9, 0.99]).tolist()} / {lengths.max()}")
    print(f"filas que quedan como {EMPTY_TOKEN}: {only_empty.mean():.1%} (vacio original: {data['is_empty'].mean():.1%})")
    print("tokens mas frecuentes:", pd.Series([t for tokens in prep for t in tokens]).value_counts().head(15).to_dict())


def main() -> None:
    show_examples()
    df = load_raw()
    for activity in ACTIVITIES:
        corpus_stats(activity, load_activity(df, activity))


if __name__ == "__main__":
    main()