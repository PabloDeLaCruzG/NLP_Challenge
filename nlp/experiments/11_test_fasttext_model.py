"""Test de validacion en Pliegue 0 de Actividad 1.
Reutiliza completamente las clases y arnes de nlp_challenge para verificar
las variantes de embedding (scratch stem/no-stem, fasttext frozen/finetune).
"""

from pathlib import Path
import sys
import time
from sklearn.model_selection import StratifiedKFold

# Asegurar importacion desde el directorio padre
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nlp_challenge import (
    load_activity,
    preprocess_corpus,
    compute_metrics,
    SeqRNNWrapper,
    SEED,
    TARGET,
)


def main() -> None:
    print("=== Test de Validacion en Pliegue 0 (Actividad 1) ===")
    df = load_activity(1)

    print("Preprocesando corpus con stemming...")
    df["tokens"] = preprocess_corpus(df["feedback"], use_stemming=True)
    print("Preprocesando corpus sin stemming (superficie)...")
    df["tokens_nostem"] = preprocess_corpus(df["feedback"], use_stemming=False)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    train_idx, test_idx = next(skf.split(df, df["stratum"]))
    train_df, test_df = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()
    y_test = test_df[TARGET].values

    experiments = [
        ("SeqRNN Scratch (Fase 2, Stem 64d)", "scratch", "tokens", 64),
        ("SeqRNN Scratch (No-Stem 64d)", "scratch", "tokens_nostem", 64),
        ("SeqRNN FastText Frozen (300d)", "fasttext_frozen", "tokens_nostem", 300),
        ("SeqRNN FastText Fine-Tuned (300d)", "fasttext_finetune", "tokens_nostem", 300),
    ]

    for name, mode, col, dim in experiments:
        t0 = time.time()
        model = SeqRNNWrapper(embedding_mode=mode, tokens_col=col, embedding_dim=dim)
        model.fit(train_df)
        preds = model.predict(test_df)
        elapsed = time.time() - t0
        metrics = compute_metrics(y_test, preds)
        print(f"\n[{name}] ({elapsed:.1f} s)")
        print(f"  MAE:  {metrics['mae']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")


if __name__ == "__main__":
    main()
