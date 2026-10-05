"""Bloque 7a: baseline constante (media y mediana) evaluado con el mismo 10-CV que usaran los modelos."""
import pandas as pd

from data import ACTIVITIES, ROOT, load_activity, load_raw
from harness import ConstantBaseline, evaluate_cv, save_results, summarize_results, wilcoxon_compare
from splits import get_or_create_splits

RESULTS_DIR = ROOT / "results"


def main() -> None:
    df = load_raw()
    for activity in ACTIVITIES:
        data = load_activity(df, activity)
        fold_id = get_or_create_splits(activity, data)
        results = pd.concat([
            evaluate_cv("const_mean", lambda: ConstantBaseline("mean"), data, fold_id),
            evaluate_cv("const_median", lambda: ConstantBaseline("median"), data, fold_id),
        ], ignore_index=True)
        save_results(results, RESULTS_DIR / f"activity{activity}_cv_results.csv")

        print(f"\n=== Actividad {activity}: baseline constante en 10-CV ===")
        print(summarize_results(results).to_string())
        year_cols = [c for c in results.columns if c.startswith("mae_")]  # mae_2015, mae_2016
        print("\nMAE medio por curso (entre pliegues):")
        print(results.groupby("model")[year_cols].mean().round(3).to_string())
        print("\nWilcoxon const_mean frente a const_median:")
        for metric in ("mae", "rmse"):
            print(" ", wilcoxon_compare(results, "const_mean", "const_median", metric))


if __name__ == "__main__":
    main()