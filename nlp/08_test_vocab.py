"""Comprobacion de min_freq en el vocabulario del pligue 0 (act 1)"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from nlp_challenge import load_activity, preprocess_corpus, assign_folds, fold_indices, SEED

# Cargar datos de la act 1
data = load_activity(1)
data["tokens"] = preprocess_corpus(data["feedback"])
# Unir los tokens en cadenas(textos preprocesados) que es lo que espera CountVectorizer
data["feedback_prep"] = preprocess_corpus(data["feedback"]).apply(lambda x: " ".join(x))

# Obtener las particiones y separar el pliegue 0
fold_id = assign_folds(data)
train_idx, text_idx = fold_indices(fold_id, 0)

train_df = data.iloc[train_idx]
X_train_full = train_df["feedback_prep"].values
y_train_full = train_df["value"].values

# Separar el 90% de entrenamiento real estratificando por stratum
X_train_fold, _, _, _ = train_test_split(
    X_train_full, y_train_full, test_size=0.10, random_state=SEED, stratify=train_df["stratum"]
)

# Probar los umbbrales
print(f"--- Analisis de Vocabulario (Pliegue 0, Act 1, 90% train = {len(X_train_fold)} filas) ---")
for freq in [1, 2, 3, 4, 5]:
    vectorizer = CountVectorizer(min_df=freq, token_pattern=r"\S+", lowercase=False)
    vectorizer.fit(X_train_fold)
    print(f"min_freq={freq} -> Tamano del vocabulario: {len(vectorizer.vocabulary_):,} palabras")
