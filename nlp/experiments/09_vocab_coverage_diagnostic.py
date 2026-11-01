"""Diagnostico de vocabulario para embeddings preentrenados.
Compara la cobertura y tamano del vocabulario con y sin stemming,
reutilizando funciones y constantes del modulo principal.
"""

from collections import Counter
from pathlib import Path
import re
import sys
import pandas as pd

# Asegurar importacion desde el directorio padre
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nlp_challenge import (
    load_activity,
    strip_accents,
    build_stopwords,
    preprocess_text,
    EMPTY_TOKEN,
)


def preprocess_corpus_variant(
    texts: pd.Series,
    use_stemming: bool = True,
    strip: bool = True
) -> pd.Series:
    """Aplica una variante del preprocesado parametrizando stemming y marcas diacriticas."""
    from nltk.stem.snowball import SnowballStemmer
    stop_set = build_stopwords()
    stemmer = SnowballStemmer("spanish") if use_stemming else None

    def process_one(text: str) -> list[str]:
        lowered = text.lower()
        if strip:
            clean = strip_accents(lowered)
            tokens = re.findall(r"[a-z0-9]+", clean)
        else:
            tokens = re.findall(r"[a-záéíóúüñ0-9]+", lowered)

        tokens = [t for t in tokens if t not in stop_set]
        if use_stemming and stemmer is not None:
            tokens = [stemmer.stem(t) for t in tokens]
        return tokens if tokens else [EMPTY_TOKEN]

    return texts.map(process_one)


def analyze_vocabulary(token_series: pd.Series, min_freq: int = 2) -> dict:
    """Calcula estadisticas de vocabulario y frecuencias."""
    counts = Counter()
    total_tokens = 0
    empty_count = 0

    for tokens in token_series:
        if tokens == [EMPTY_TOKEN]:
            empty_count += 1
        total_tokens += len(tokens)
        counts.update(tokens)

    return {
        "total_tokens": total_tokens,
        "unique_tokens": len(counts),
        "frequent_tokens_ge2": sum(1 for w, c in counts.items() if c >= min_freq and w != EMPTY_TOKEN),
        "empty_samples": empty_count,
        "empty_ratio": empty_count / len(token_series),
        "counts": counts
    }


def main() -> None:
    for activity in [1, 2]:
        df = load_activity(activity)
        print(f"\n=======================================================")
        print(f" ACTIVIDAD {activity} ({len(df):,} muestras)")
        print(f"=======================================================")

        # Variante 1: Fase 2 (sin acentos, con stemming)
        v1 = preprocess_corpus_variant(df["feedback"], use_stemming=True, strip=True)
        stats_v1 = analyze_vocabulary(v1)

        # Variante 2: Superficie sin acentos (sin stemming)
        v2 = preprocess_corpus_variant(df["feedback"], use_stemming=False, strip=True)
        stats_v2 = analyze_vocabulary(v2)

        # Variante 3: Superficie con acentos (sin stemming)
        v3 = preprocess_corpus_variant(df["feedback"], use_stemming=False, strip=False)
        stats_v3 = analyze_vocabulary(v3)

        print("\n--- Comparativa de Vocabularios ---")
        print(f"1. Fase 2 (Stemming, sin acentos):")
        print(f"   Tokens unicos: {stats_v1['unique_tokens']:,} | Frecuencia >= 2: {stats_v1['frequent_tokens_ge2']:,} | Vacio: {stats_v1['empty_ratio']:.2%}")

        print(f"\n2. Superficie (Sin stemming, sin acentos):")
        print(f"   Tokens unicos: {stats_v2['unique_tokens']:,} | Frecuencia >= 2: {stats_v2['frequent_tokens_ge2']:,} | Vacio: {stats_v2['empty_ratio']:.2%}")

        print(f"\n3. Superficie original (Sin stemming, CON acentos):")
        print(f"   Tokens unicos: {stats_v3['unique_tokens']:,} | Frecuencia >= 2: {stats_v3['frequent_tokens_ge2']:,} | Vacio: {stats_v3['empty_ratio']:.2%}")


if __name__ == "__main__":
    main()
