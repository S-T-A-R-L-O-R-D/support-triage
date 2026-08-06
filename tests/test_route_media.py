"""Part C: screenshots route into the same four queues via cached OCR text.

The committed OCR cache makes these tests (and the CLI) runnable without
the optional OCR dependency — the cache is the documented fallback.
"""
import csv

from triage.classifier import REPO_ROOT
from triage.route_media import main

EXPECTED = {
    "login-error.png": "account-access",
    "phishing-sms.png": "fraud-report",
    "txn-failed.png": "transaction-dispute",
}


def test_screenshots_route_to_their_queues_from_cache(tmp_path):
    out = tmp_path / "routes.csv"
    main(["--input", str(REPO_ROOT / "media" / "screenshots"), "--output", str(out)])

    rows = {r["file"]: r for r in csv.DictReader(open(out, newline="", encoding="utf-8"))}
    assert {f: r["route"] for f, r in rows.items()} == EXPECTED
    assert all(r["source"] == "cache" for r in rows.values())


def test_confidence_is_reported_for_review_triage(tmp_path):
    out = tmp_path / "routes.csv"
    main(["--input", str(REPO_ROOT / "media" / "screenshots"), "--output", str(out)])
    rows = list(csv.DictReader(open(out, newline="", encoding="utf-8")))
    assert all(0.0 < float(r["confidence"]) <= 1.0 for r in rows)
    assert all(r["needs_review"] in {"yes", "no"} for r in rows)
