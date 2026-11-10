"""Train the SMS spam classification experiments and save reproducible artifacts."""
import json
import time
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.data_loader import load_dataset
from src.preprocessing import build_experiments

ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def run_training_pipeline(save_models=True):
    """Train all experiment pipelines and optionally save models and predictions."""
    print("=== [STEP 1] Load and prepare dataset ===")
    df = load_dataset(ROOT_DIR / "data" / "SMSSpamCollection")

    X = df["text"]
    y = df["label"]
    total_samples = len(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    print(
        f"Samples: {total_samples} | Train: {len(X_train)} "
        f"| Test: {len(X_test)}"
    )

    experiments = build_experiments(random_state=RANDOM_STATE)
    trained_models = {}
    predictions = {}
    metrics = {}

    print("\n=== [STEP 2] Train models ===")
    for experiment_name, pipeline in experiments.items():
        start_time = time.perf_counter()
        pipeline.fit(X_train, y_train)
        elapsed = time.perf_counter() - start_time

        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        trained_models[experiment_name] = pipeline
        predictions[experiment_name] = [str(value) for value in y_pred]
        metrics[experiment_name] = {
            "accuracy": float(accuracy),
            "train_time_seconds": float(elapsed),
        }

        print(
            f"{experiment_name:14s} | Accuracy: {accuracy:.4%} "
            f"| Time: {elapsed:.4f}s"
        )

    if save_models:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        for experiment_name, model in trained_models.items():
            joblib.dump(model, MODELS_DIR / f"{experiment_name}.joblib")

        # Store plain Python lists to avoid coupling the artifact to pandas extension dtypes.
        artifacts = {
            "X_test": [str(value) for value in X_test],
            "y_test": [str(value) for value in y_test],
            "predictions": predictions,
        }
        joblib.dump(artifacts, RESULTS_DIR / "predictions_and_test.joblib")

        summary = {
            "data_summary": {
                "total_samples": int(total_samples),
                "train_samples": int(len(X_train)),
                "test_samples": int(len(X_test)),
                "random_state": RANDOM_STATE,
                "test_size": TEST_SIZE,
                "label_order": ["ham", "spam"],
            },
            "experiments": metrics,
        }
        with (RESULTS_DIR / "training_summary.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(summary, file, indent=4, ensure_ascii=False)

        print(f"Models saved to: {MODELS_DIR}")
        print(f"Training artifacts saved to: {RESULTS_DIR}")

    return metrics, {
        "models": trained_models,
        "X_test": X_test,
        "y_test": y_test,
        "predictions": predictions,
    }


if __name__ == "__main__":
    run_training_pipeline(save_models=True)
