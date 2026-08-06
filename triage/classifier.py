"""Route classifier: the handover model behind a clean train/predict API.

The TF-IDF + logistic-regression model from the baseline was fine once
honestly evaluated (see README, Part A), so production uses the same one,
trained on the full labelled set. Defined here rather than imported from
baseline/ so the handed-over script can be deleted without breaking prod.
"""
import csv
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_CSV = REPO_ROOT / "data" / "train.csv"
ROUTES = ["account-access", "transaction-dispute", "fraud-report", "general"]


def load_labelled(path: str | Path) -> tuple[list[str], list[str]]:
    """Read a text,label CSV, failing loudly on a malformed file."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or not {"text", "label"} <= set(reader.fieldnames):
            raise ValueError(f"{path}: expected columns 'text' and 'label'")
        rows = [r for r in reader if (r["text"] or "").strip()]
    if not rows:
        raise ValueError(f"{path}: no usable rows")
    unknown = {r["label"] for r in rows} - set(ROUTES)
    if unknown:
        raise ValueError(f"{path}: unknown labels {sorted(unknown)}")
    return [r["text"] for r in rows], [r["label"] for r in rows]


def build_model() -> Pipeline:
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True),
        LogisticRegression(max_iter=2000, C=10.0),
    )


def train(train_csv: str | Path = DEFAULT_TRAIN_CSV) -> Pipeline:
    texts, labels = load_labelled(train_csv)
    return build_model().fit(texts, labels)
