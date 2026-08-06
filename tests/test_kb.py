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


def test_datetime_with_time_component_is_rejected():
    raw = (
        "---\ndoc_id: t1\ntitle: T\neffective_date: 2026-01-01 09:00:00\n---\nbody"
    )
    with pytest.raises(ValueError, match="date"):
        parse_doc(raw, source="t1.md")


def _write_doc(path, doc_id, superseded_by="", supersedes=""):
    path.write_text(
        f"---\ndoc_id: {doc_id}\ntitle: T\neffective_date: 2026-01-01\n"
        f"superseded_by: {superseded_by}\nsupersedes: {supersedes}\n---\nbody",
        encoding="utf-8",
    )


def test_dangling_superseded_by_link_is_rejected_at_load(tmp_path):
    _write_doc(tmp_path / "a.md", "a", superseded_by="ghost")
    with pytest.raises(ValueError, match="ghost"):
        load_kb(tmp_path)


def test_asymmetric_supersedes_links_are_rejected_at_load(tmp_path):
    # b claims to supersede a, but a does not point back — the silent
    # split-chain case where a stale doc would stay "in force".
    _write_doc(tmp_path / "a.md", "a")
    _write_doc(tmp_path / "b.md", "b", supersedes="a")
    with pytest.raises(ValueError, match="disagree"):
        load_kb(tmp_path)


def test_kb_file_with_utf8_bom_still_loads(tmp_path):
    _write_doc(tmp_path / "a.md", "a")
    raw = (tmp_path / "a.md").read_text(encoding="utf-8")
    (tmp_path / "a.md").write_text(raw, encoding="utf-8-sig")
    assert load_kb(tmp_path)[0].doc_id == "a"
