"""Train and evaluate baseline and comparison models."""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
)

try:
    from .data_loader import load_dataset
    from .preprocessing import build_experiments
except ImportError:  # supports `python src/train.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.data_loader import load_dataset
    from src.preprocessing import build_experiments

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()

    # Fixed hold-out split; stratify preserves the original class ratio.
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.20,
        random_state=42,
        stratify=df["label"],
    )

    metrics_rows = []
    for name, model in build_experiments().items():
        print(f"\n=== {name} ===")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        metrics_rows.append({
            "experiment": name,
            "accuracy": accuracy_score(y_test, predictions),
            "spam_precision": precision_score(
                y_test, predictions, pos_label="spam", zero_division=0
            ),
            "spam_recall": recall_score(
                y_test, predictions, pos_label="spam", zero_division=0
            ),
            "spam_f1": f1_score(
                y_test, predictions, pos_label="spam", zero_division=0
            ),
        })

        print(classification_report(
            y_test, predictions, labels=["ham", "spam"], zero_division=0
        ))

        cm = confusion_matrix(y_test, predictions, labels=["ham", "spam"])
        display = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=["ham", "spam"]
        )
        display.plot(values_format="d", cmap="Blues")
        plt.title(f"Confusion Matrix — {name}")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"confusion_matrix_{name}.png", dpi=160)
        plt.close()

        error_df = pd.DataFrame({
            "text": X_test.reset_index(drop=True),
            "true_label": y_test.reset_index(drop=True),
            "predicted_label": pd.Series(predictions),
        })
        error_df = error_df[error_df["true_label"] != error_df["predicted_label"]]
        error_df.to_csv(
            RESULTS_DIR / f"misclassified_{name}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(RESULTS_DIR / "metrics.csv", index=False, encoding="utf-8-sig")
    print("\n=== Tổng hợp metrics ===")
    print(metrics.to_string(index=False))
    print(f"\nĐã lưu kết quả tại: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
