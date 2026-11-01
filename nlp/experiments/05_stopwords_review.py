"""
Bloque 5a: revision de stopwords

No usa la nota (value): la lista de palabras a conservar se decide por criterio linguistico y se
validara despues dentro del 10-CV
"""
import re
import unicodedata
from collections import Counter

import nltk
import pandas as pd
from nltk.corpus import stopwords

from data import ACTIVITIES, load_activity, load_raw


def strip_accents(text: str) -> str:
    """Quita los acentos, por ejemplo 'esta' con y sin acento pasa a ser la misma cadena"""
    decomposed = unicodedata.normalize("NFD", text)  # separa cada letra de su marca de acento
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")  # Mn = marca: se descarta


def tokenize(text: str) -> list[str]:
    """Minusculas, sin acentos y solo secuencias de letras o digitos"""
    return re.findall(r"[a-z0-9]+", strip_accents(text.lower()))


def load_stopwords() -> set[str]:
    """Lista estandar de NLTK en espanol"""
    nltk.download("stopwords", quiet=True)
    return set(stopwords.words("spanish"))


def accent_report(raw_stopwords: set[str]) -> None:
    """Cuantas stopwords llevan acento y cuantas dejarian de coincidir si solo quita acentos al texto"""
    accented = [w for w in raw_stopwords if strip_accents(w) != w]
    lost = sorted(w for w in accented if strip_accents(w) not in raw_stopwords)
    print(f"lista estandar: {len(raw_stopwords)} palabras | con acento: {len(accented)}")
    print(f"con acento y SIN su version sin acento en la lista ({len(lost)}): {lost[:15]}")


def stopword_frequency(texts: pd.Series, stop_norm: set[str], top: int = 50) -> pd.DataFrame:
    """Apariciones totales de cada stopword y porcentaje de filas que la contienen"""
    counts, rows_with = Counter(), Counter()
    for text in texts:
        found = [t for t in tokenize(text) if t in stop_norm]
        counts.update(found)
        rows_with.update(set(found))  # set: cuenta una sola vez por fila
    table = pd.DataFrame({"count": counts, "pct_rows": {w: 100 * n / len(texts) for w, n in rows_with.items()}})
    return table.sort_values("count", ascending=False).head(top).round(1)


def main() -> None:
    raw_stopwords = load_stopwords()
    accent_report(raw_stopwords)
    stop_norm = {strip_accents(w) for w in raw_stopwords}  # la lista pasa por la misma normalizacion que el texto
    df = load_raw()
    corpus = pd.concat([load_activity(df, a) for a in ACTIVITIES])
    print(f"\ntextos: {len(corpus):,} filas (actividades 1 y 2, cursos 2015 y 2016)")
    print(stopword_frequency(corpus["feedback"], stop_norm).to_string())


if __name__ == "__main__":
    main()