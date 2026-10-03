"""
Pipeline de preprocesado de texto del desafio NLP

Es un pipeline SIN ESTADO: no aprende nada de los datos (ni vocabulario ni frecuencias), asi que
aplicarlo a todo el corpus antes de partir en pliegues no produce fuga de datos.
Lo que si aprende (vocabulario, IDF) se construira dentro de cada pliegue mas adelante.
"""
import re
import unicodedata

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

EMPTY_TOKEN = "<empty>"  # texto que se queda sin ningun token (vacio real o solo signos)

# Stopwords que se conservan
KEEP_STOPWORDS = {
    "no",
    "nada",
    "ni",
    "sin",
    "muy",
    "mas",
    "mucho",
    "hay",
    "tiene",
    "poco",
    "tanto",
    "algo",
    "todo",
    "todos",
    "pero",
    "si",
}


def strip_accents(text: str) -> str:
    """Quita los acentos"""
    decomposed = unicodedata.normalize("NFD", text)  # separa cada letra de su marca de acento
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")  # Mn = marca: se descarta


def tokenize(text: str) -> list[str]:
    """Minusculas, sin acentos y solo secuencias de letras o digitos"""
    return re.findall(r"[a-z0-9]+", strip_accents(text.lower()))


def build_stopwords(keep: set[str] = KEEP_STOPWORDS) -> set[str]:
    """Lista de NLTK normalizada igual que el texto (sin acentos) y sin las palabras a conserva"""
    nltk.download("stopwords", quiet=True)
    normalized = {strip_accents(w) for w in stopwords.words("spanish")}
    return normalized - {strip_accents(w) for w in keep}  # la lista a conservar tambien se normaliza


def preprocess_text(text: str, stop_set: set[str], stemmer: SnowballStemmer) -> list[str]:
    """minusculas -> sin acentos -> tokens -> sin stopwords -> stemming, y si no queda nada, <empty>"""
    tokens = [t for t in tokenize(text) if t not in stop_set]
    stems = [stemmer.stem(t) for t in tokens]
    return stems if stems else [EMPTY_TOKEN]  # cada secuencia tendra al menos un token


def preprocess_corpus(texts: pd.Series, keep: set[str] = KEEP_STOPWORDS) -> pd.Series:
    """Aplica el pipeline a una serie de textos y devuelve una serie de listas de tokens"""
    stop_set = build_stopwords(keep)
    stemmer = SnowballStemmer("spanish")
    return texts.map(lambda text: preprocess_text(text, stop_set, stemmer))
