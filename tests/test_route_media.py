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


def test_cache_is_keyed_by_content_not_just_filename(tmp_path):
    from triage.route_media import _cache_path

    a, b = tmp_path / "x" / "shot.png", tmp_path / "y" / "shot.png"
    a.parent.mkdir()
    b.parent.mkdir()
    a.write_bytes(b"image-one")
    b.write_bytes(b"image-two")
    cache = tmp_path / "cache"
    assert _cache_path(a, cache) != _cache_path(b, cache)


def test_confidence_is_reported_for_review_triage(tmp_path):
    out = tmp_path / "routes.csv"
    main(["--input", str(REPO_ROOT / "media" / "screenshots"), "--output", str(out)])
    rows = list(csv.DictReader(open(out, newline="", encoding="utf-8")))
    assert all(0.0 < float(r["confidence"]) <= 1.0 for r in rows)
    assert all(r["needs_review"] in {"yes", "no"} for r in rows)
