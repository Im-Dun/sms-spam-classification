
from pathlib import Path
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# Project paths
ROOT_DIR = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT_DIR / "results" / "predictions_and_test.joblib"
RESULTS_DIR = ROOT_DIR / "results"

LABELS = ["ham", "spam"]
POSITIVE_LABEL = "spam"


def evaluate_models():
    """Evaluate trained models using saved test predictions."""

    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError(
            f"Prediction artifact not found: {ARTIFACT_PATH}"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    artifacts = joblib.load(ARTIFACT_PATH)

    X_test = artifacts["X_test"]
    y_test = artifacts["y_test"]
    predictions = artifacts["predictions"]

    metrics = []
    summary = {}

    for model_name, y_pred in predictions.items():
        if len(y_pred) != len(y_test):
            raise ValueError(
                f"{model_name}: prediction length does not match y_test"
            )

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(
            y_test, y_pred, pos_label=POSITIVE_LABEL, zero_division=0
        )
        recall = recall_score(
            y_test, y_pred, pos_label=POSITIVE_LABEL, zero_division=0
        )
        f1 = f1_score(
            y_test, y_pred, pos_label=POSITIVE_LABEL, zero_division=0
        )

        # Confusion matrix: [[TN, FP], [FN, TP]]
        cm = confusion_matrix(y_test, y_pred, labels=LABELS)
        tn, fp, fn, tp = cm.ravel()

        metrics.append({
            "model": model_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
        })

        summary[model_name] = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "confusion_matrix": cm.tolist(),
        }

        # Save confusion matrix
        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=LABELS,
        )
        display.plot(cmap="Blues", values_format="d")
        plt.title(f"Confusion Matrix - {model_name}")
        plt.tight_layout()
        plt.savefig(
            RESULTS_DIR / f"confusion_matrix_{model_name}.png",
            dpi=300,
        )
        plt.close()

        # Save misclassified messages
        X_values = list(X_test)
        y_values = list(y_test)
        y_predictions = list(y_pred)

        errors = []

        for text, actual, predicted in zip(
            X_values, y_values, y_predictions
        ):
            if actual != predicted:
                error_type = (
                    "FP" if predicted == POSITIVE_LABEL else "FN"
                )

                errors.append({
                    "text": text,
                    "actual": actual,
                    "predicted": predicted,
                    "error_type": error_type,
                })

        pd.DataFrame(errors).to_csv(
            RESULTS_DIR / f"misclassified_{model_name}.csv",
            index=False,
            encoding="utf-8-sig",
        )

        print(
            f"{model_name}: "
            f"Accuracy={accuracy:.4f}, "
            f"Precision={precision:.4f}, "
            f"Recall={recall:.4f}, "
            f"F1={f1:.4f}"
        )

    # Export comparison table
    metrics_df = pd.DataFrame(metrics)
    metrics_df = metrics_df.sort_values(
        by="f1_score", ascending=False
    )
    metrics_df.to_csv(
        RESULTS_DIR / "metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Export evaluation summary
    with open(
        RESULTS_DIR / "evaluation_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=4, ensure_ascii=False)

    # Compare F1 scores
    plt.figure(figsize=(9, 5))
    plt.bar(metrics_df["model"], metrics_df["f1_score"])
    plt.ylabel("F1-score")
    plt.xlabel("Model")
    plt.title("F1-score Comparison")
    plt.ylim(0, 1)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "f1_comparison.png",
        dpi=300,
    )
    plt.close()

    print("\n=== Evaluation completed ===")
    print(metrics_df.to_string(index=False))
    print(f"\nResults saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    evaluate_models()
