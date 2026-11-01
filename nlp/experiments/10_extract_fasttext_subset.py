"""Descarga y extraccion de vectores FastText en castellano.
Reutiliza funciones y constantes de nlp_challenge para extraer unicamente los vectores
de las palabras del corpus sin redundancias de codigo.
"""

from collections import Counter
import gzip
from pathlib import Path
import sys
import time
import urllib.request

import numpy as np
import torch

# Asegurar importacion desde el directorio padre
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nlp_challenge import (
    load_activity,
    strip_accents,
    preprocess_corpus,
    FASTTEXT_SUBSET_PATH,
)

FASTTEXT_DIR = Path("data/fasttext")
LOCAL_GZ_PATH = FASTTEXT_DIR / "cc.es.300.vec.gz"
FASTTEXT_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.es.300.vec.gz"


def extract_candidate_vocabulary() -> tuple[set[str], dict[str, int]]:
    """Extrae la union de palabras candidatas de ambas actividades sin stemming."""
    word_counts = Counter()
    for act in [1, 2]:
        df = load_activity(act)
        # Preprocesar corpus sin stemming para conservar palabras de superficie
        tokens_series = preprocess_corpus(df["feedback"], use_stemming=False)
        for tokens in tokens_series:
            word_counts.update(tokens)

    target_words = set(word_counts.keys())
    print(f"Total de palabras unicas en el corpus: {len(target_words):,}")
    print(f"Palabras con frecuencia >= 2: {sum(1 for c in word_counts.values() if c >= 2):,}")
    return target_words, dict(word_counts)


def download_fasttext_gz() -> Path:
    """Descarga el archivo comprimido oficial si no existe localmente."""
    FASTTEXT_DIR.mkdir(parents=True, exist_ok=True)
    if LOCAL_GZ_PATH.exists() and LOCAL_GZ_PATH.stat().st_size > 1_000_000_000:
        print(f"Archivo FastText encontrado en: {LOCAL_GZ_PATH} ({LOCAL_GZ_PATH.stat().st_size / 1e9:.2f} GB)")
        return LOCAL_GZ_PATH

    print(f"Descargando FastText en castellano desde:\n{FASTTEXT_URL}")
    start_time = time.time()

    def reporthook(count, block_size, total_size):
        if count % 2000 == 0:
            downloaded_mb = count * block_size / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            pct = 100 * downloaded_mb / total_mb if total_size > 0 else 0
            elapsed = time.time() - start_time
            speed = downloaded_mb / elapsed if elapsed > 0 else 0
            print(f"  Descargado: {downloaded_mb:.1f}/{total_mb:.1f} MB ({pct:.1f}%) a {speed:.2f} MB/s", end="\r")

    urllib.request.urlretrieve(FASTTEXT_URL, LOCAL_GZ_PATH, reporthook=reporthook)
    print(f"\nDescarga completada en {time.time() - start_time:.1f} segundos.")
    return LOCAL_GZ_PATH


def stream_extract_vectors(gz_path: Path, target_words: set[str]) -> tuple[dict[str, np.ndarray], int]:
    """Lee el archivo gz linea por linea en streaming y extrae solo las palabras deseadas."""
    print("Iniciando extraccion de vectores en streaming...")
    found_vectors = {}
    total_in_fasttext = 0
    vector_dim = 300

    start_time = time.time()
    with gzip.open(gz_path, "rt", encoding="utf-8", errors="ignore") as f:
        header = f.readline().strip().split()
        if len(header) == 2:
            total_in_fasttext = int(header[0])
            vector_dim = int(header[1])

        lines_processed = 0
        for line in f:
            lines_processed += 1
            if lines_processed % 250_000 == 0:
                elapsed = time.time() - start_time
                print(f"  Lineas procesadas: {lines_processed:,}/{total_in_fasttext:,} | "
                      f"Vectores encontrados: {len(found_vectors):,}/{len(target_words):,} "
                      f"({elapsed:.1f} s)", end="\r")

            parts = line.rstrip().split(" ")
            word_clean = strip_accents(parts[0].lower())

            if word_clean in target_words and word_clean not in found_vectors:
                try:
                    vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                    if len(vec) == vector_dim:
                        found_vectors[word_clean] = vec
                except ValueError:
                    continue

            if len(found_vectors) == len(target_words):
                break

    print(f"\nExtraccion completada: {len(found_vectors):,} vectores encontrados ({len(found_vectors)/len(target_words):.2%}).")
    return found_vectors, vector_dim


def save_compact_subset(vectors: dict[str, np.ndarray], word_counts: dict[str, int], dim: int) -> None:
    """Guarda los vectores y metadatos en un fichero PyTorch compacto."""
    torch_vectors = {w: torch.from_numpy(v) for w, v in vectors.items()}
    payload = {
        "embedding_dim": dim,
        "n_words": len(vectors),
        "vectors": torch_vectors,
        "word_counts": word_counts
    }
    FASTTEXT_SUBSET_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, FASTTEXT_SUBSET_PATH)
    print(f"Subconjunto guardado en: {FASTTEXT_SUBSET_PATH} ({FASTTEXT_SUBSET_PATH.stat().st_size / 1e6:.2f} MB)")


def main() -> None:
    print("=== Extraccion de Subconjunto de Vectores FastText ===")
    target_words, word_counts = extract_candidate_vocabulary()
    gz_path = download_fasttext_gz()
    vectors, dim = stream_extract_vectors(gz_path, target_words)
    save_compact_subset(vectors, word_counts, dim)


if __name__ == "__main__":
    main()
