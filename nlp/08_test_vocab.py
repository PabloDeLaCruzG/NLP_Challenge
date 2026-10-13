"""Comprobacion de min_freq en el vocabulario del pligue 0 (act 1)"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from nlp_challenge import load_activity, preprocess_corpus, assign_folds, fold_indices

# Cargar datos de la act 1
data = load_activity(1)
# Unir los tokens en cadenas(textos preprocesados) que es lo que espera CountVectorizer
data["feedback_prep"] = preprocess_corpus(data["feedback"]).apply(lambda x: " ".join(x))

# Obtener las particiones y separar el pliegue 0
fold_id = assign_folds(data)
train_idx, text_idx = fold_indices(fold_id, 0)

X_train_full = data.iloc[train_idx]["feedback_prep"].values
y_train_full = data.iloc[train_idx]["value"].values

# Extraer el 10% para validacion interna
X_train_fold, X_val_internal, _, _ = train_test_split(
    X_train_full, y_train_full, test_size=0.10, random_state=42
)

# Probar los umbbrales
print("ANALISIS de VOCABULARIO (PLIEGUE 0, Act 1)")
for freq in [1, 2, 3, 4, 5]:
    vectorizer = CountVectorizer(min_df=freq)
    vectorizer.fit(X_train_fold)
    print(f"min_freq={freq} , Tamaño del vocabulario: {len(vectorizer.vocabulary_):,} palabras")
