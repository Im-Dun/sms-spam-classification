"""Text vectorizers and model definitions for the SMS spam experiments."""
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_experiments(random_state: int = 42):
    """Return named sklearn Pipelines. Vectorizers are fitted only on training data."""
    return {
        "bow_nb": Pipeline([
            ("vectorizer", CountVectorizer()),
            ("classifier", MultinomialNB()),
        ]),
        "tfidf_nb": Pipeline([
            ("vectorizer", TfidfVectorizer()),
            ("classifier", MultinomialNB()),
        ]),
        "tfidf_logreg": Pipeline([
            ("vectorizer", TfidfVectorizer()),
            ("classifier", LogisticRegression(
                max_iter=1000,
                class_weight=None,
                random_state=random_state,
            )),
        ]),
    }
