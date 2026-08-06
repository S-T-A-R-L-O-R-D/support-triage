"""Support-ticket route classifier, baseline.

Handed over from a previous engineer along with eval_report.md.
Trains on data/train.csv and reports how well it does.

    python3 baseline/baseline_classifier.py

Review note (Part A): the original evaluation random-split 400 rows of
which 262 are boilerplate variants of another row ("Urgent: X" vs
"Hey, X. Thanks."), so 49 of its 80 test rows had a near-twin in train,
and the vectorizer was fitted on the full corpus before splitting. Both
leak test information into training and inflated accuracy to 98.75%.
The evaluation below keeps variants of one message on the same side of
the split, fits the vectorizer inside each fold, and reports per-route
metrics, because the four routes are imbalanced and fraud-report is the
expensive one to miss. Verdict and numbers: README, Part A.
"""
import csv
import os
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "train.csv")
ROUTES = ["account-access", "transaction-dispute", "fraud-report", "general"]

# Greeting/urgency boilerplate that the message templates wrap around a core
# sentence. Rows sharing a core are the same message and must not be split
# across train and test.
_PREFIX = re.compile(
    r"^(urgent:|please help\.|quick question,|hello team,|hey,|hi,)\s*", re.I
)
_SUFFIX = re.compile(
    r"\s*(thanks\.|please advise\.|let me know\.|appreciate any help\.|"
    r"this is time sensitive\.)$",
    re.I,
)


def core(text):
    """Strip boilerplate wrappers so variants of one message share a key."""
    t = text.strip().lower()
    while True:
        stripped = _PREFIX.sub("", _SUFFIX.sub("", t)).strip()
        if stripped == t:
            return t
        t = stripped


def build_model():
    """TF-IDF + logistic regression, unchanged from the handover.

    The model was never the problem; the evaluation was.
    """
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True),
        LogisticRegression(max_iter=2000, C=10.0),
    )


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r["text"] for r in rows], [r["label"] for r in rows]


def main():
    texts, labels = load(DATA)
    print(f"loaded {len(texts)} rows")

    X = np.array(texts, dtype=object)
    y = np.array(labels)
    groups = np.array([core(t) for t in texts])
    print(f"{len(set(groups))} unique message cores after stripping boilerplate")

    # 5-fold CV, stratified by route, grouped by core: every row is scored
    # exactly once, never by a model that saw a variant of it.
    fold_accs = []
    preds = np.empty_like(y)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    for train_idx, test_idx in cv.split(X, y, groups):
        model = build_model().fit(X[train_idx], y[train_idx])
        preds[test_idx] = model.predict(X[test_idx])
        fold_accs.append(accuracy_score(y[test_idx], preds[test_idx]))

    acc = float(np.mean(fold_accs))
    print(f"accuracy: {acc:.3f} +/- {np.std(fold_accs):.3f} across 5 grouped folds")
    print(classification_report(y, preds, digits=3))
    print("confusion matrix (rows = true route):")
    print(ROUTES)
    print(confusion_matrix(y, preds, labels=ROUTES))
    return acc


_fitted = None


def predict(text):
    """predict(text) -> route label.

    The handover version re-trained the model on every call; fit once and
    reuse it.
    """
    global _fitted
    if _fitted is None:
        texts, labels = load(DATA)
        _fitted = build_model().fit(texts, labels)
    return _fitted.predict([text])[0]


if __name__ == "__main__":
    main()
