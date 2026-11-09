"""Text vectorizers and model definitions for the SMS spam experiments."""
import re
from collections.abc import Iterable

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

_SMS_TOKEN_PATTERN = re.compile(
    r"https?://[^\s]+|www\.[^\s]+|[£$€]|[\w]+(?:['’][\w]+)*|[^\w\s]+",
    flags=re.IGNORECASE,
)


def normalize_sms_text(text: str) -> str:
    """Collapse excess whitespace and lowercase an SMS without deleting its content."""
    if not isinstance(text, str):
        raise TypeError("SMS text must be a string.")
    return " ".join(text.split()).lower()


def normalize_sms_texts(texts: Iterable[str]) -> list[str]:
    """Validate a batch of SMS texts and normalize each one."""
    if isinstance(texts, (str, bytes)):
        raise TypeError("Expected an iterable of SMS text strings, not a single string.")

    normalized_texts = []
    for index, text in enumerate(texts):
        if not isinstance(text, str):
            raise ValueError(f"SMS text at position {index} must be a string.")
        normalized_text = normalize_sms_text(text)
        if not normalized_text:
            raise ValueError(f"SMS text at position {index} must not be empty.")
        normalized_texts.append(normalized_text)

    if not normalized_texts:
        raise ValueError("At least one SMS text is required.")
    return normalized_texts


def tokenize_sms(text: str) -> list[str]:
    """Extract words, numbers, URLs, currency symbols, and punctuation."""
    return _SMS_TOKEN_PATTERN.findall(text)


def _make_text_pipeline(
    vectorizer: type[CountVectorizer] | type[TfidfVectorizer],
    classifier: MultinomialNB | LogisticRegression,
) -> Pipeline:
    """Build a training-safe pipeline for raw SMS text."""
    return Pipeline([
        ("preprocessor", FunctionTransformer(normalize_sms_texts, validate=False)),
        ("vectorizer", vectorizer(
            tokenizer=tokenize_sms,
            token_pattern=None,
            lowercase=False,
            ngram_range=(1, 2),
            min_df=1,
        )),
        ("classifier", classifier),
    ])


def build_experiments(random_state: int = 42) -> dict[str, Pipeline]:
    """Return named BoW and TF-IDF pipelines for ``X_train, y_train``."""
    return {
        "bow_nb": _make_text_pipeline(CountVectorizer, MultinomialNB()),
        "tfidf_nb": _make_text_pipeline(TfidfVectorizer, MultinomialNB()),
        "tfidf_logreg": _make_text_pipeline(
            TfidfVectorizer,
            LogisticRegression(
                max_iter=1000,
                class_weight=None,
                random_state=random_state,
            ),
        ),
    }
