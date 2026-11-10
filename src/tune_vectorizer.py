
"""Compare vectorizer configurations for SMS spam classification."""

import json
from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.data_loader import load_dataset
from src.preprocessing import normalize_sms_texts, tokenize_sms

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "SMSSpamCollection"
OUTPUT_PATH = ROOT_DIR / "results" / "vectorizer_experiments.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2


def build_pipeline(ngram_range, min_df, max_df):
    """Build a BoW + MultinomialNB pipeline with selected parameters."""
    return Pipeline([
        (
            "preprocessor",
            FunctionTransformer(normalize_sms_texts, validate=False),
        ),
        (
            "vectorizer",
            CountVectorizer(
                tokenizer=tokenize_sms,
                token_pattern=None,
                lowercase=False,
                ngram_range=ngram_range,
                min_df=min_df,
                max_df=max_df,
            ),
        ),
        ("classifier", MultinomialNB()),
    ])


def main():
    df = load_dataset(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=TEST_SIZE,
        stratify=df["label"],
        random_state=RANDOM_STATE,
    )

    configurations = [
        {"name": "unigram", "ngram_range": (1, 1), "min_df": 1, "max_df": 1.0},
        {"name": "unigram_bigram", "ngram_range": (1, 2), "min_df": 1, "max_df": 1.0},
        {"name": "min_df_2", "ngram_range": (1, 2), "min_df": 2, "max_df": 1.0},
        {"name": "min_df_3", "ngram_range": (1, 2), "min_df": 3, "max_df": 1.0},
        {"name": "max_df_0.95", "ngram_range": (1, 2), "min_df": 1, "max_df": 0.95},
    ]

    results = []

    print("=== Vectorizer experiments ===")

    for config in configurations:
        model = build_pipeline(
            config["ngram_range"],
            config["min_df"],
            config["max_df"],
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        metrics = {
            "name": config["name"],
            "ngram_range": list(config["ngram_range"]),
            "min_df": config["min_df"],
            "max_df": config["max_df"],
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(
                y_test, predictions, pos_label="spam", zero_division=0
            ),
            "recall": recall_score(
                y_test, predictions, pos_label="spam", zero_division=0
            ),
            "f1": f1_score(
                y_test, predictions, pos_label="spam", zero_division=0
            ),
        }

        results.append(metrics)

        print(
            f"{config['name']:16s} | "
            f"Accuracy: {metrics['accuracy']:.4%} | "
            f"Precision: {metrics['precision']:.4%} | "
            f"Recall: {metrics['recall']:.4%} | "
            f"F1: {metrics['f1']:.4%}"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)

    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
