"""KB loading and temporal validity.

The withdrawal-fee chain kb-011 -> kb-012 -> kb-013 is the canonical case:
three versions of one policy, only one valid on any given date. `status`
says what was true when the pack was generated; the dates decide validity
for an arbitrary as_of, so a "superseded" doc must win for a historical
question that falls inside its window.
"""
from datetime import date

import pytest

from triage.kb import KB_DIR, docs_valid_at, load_kb, parse_doc


@pytest.fixture(scope="session")
def kb():
    return load_kb(KB_DIR)


def test_loads_all_docs_with_unique_ids(kb):
    assert len(kb) == 31
    assert len({d.doc_id for d in kb}) == 31


def test_historical_date_selects_the_superseded_version(kb):
    fee_docs = {d.doc_id: d for d in kb if d.doc_id in {"kb-011", "kb-012", "kb-013"}}
    on = date(2025, 6, 1)  # q11's as_of: the 0.9% era
    assert not fee_docs["kb-011"].valid_at(on)
    assert fee_docs["kb-012"].valid_at(on)
    assert not fee_docs["kb-013"].valid_at(on)  # effective 2026-05-01, still future


def test_today_selects_the_current_version(kb):
    valid_ids = {d.doc_id for d in docs_valid_at(kb, date(2026, 7, 28))}
    assert "kb-013" in valid_ids
    assert {"kb-011", "kb-012"} & valid_ids == set()


def test_expired_notice_is_only_valid_inside_its_window(kb):
    notice = next(d for d in kb if d.doc_id == "kb-091")  # June maintenance
    assert notice.valid_at(date(2026, 6, 13))
    assert not notice.valid_at(date(2026, 7, 28))


def test_open_ended_doc_has_no_expiry(kb):
    tax = next(d for d in kb if d.doc_id == "kb-101")
    assert tax.valid_until is None
    assert tax.valid_at(date(2030, 1, 1))


def test_malformed_frontmatter_is_rejected():
    with pytest.raises(ValueError, match="effective_date"):
        parse_doc("---\ndoc_id: broken\n---\nbody", source="broken.md")
