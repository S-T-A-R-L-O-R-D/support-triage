"""Knowledge-base loading and temporal validity.

A document answers for a date `as_of` when
`effective_date <= as_of <= valid_until` (open-ended if valid_until is
empty). The `status` field is deliberately not used for selection: it
describes the moment the KB snapshot was taken, while the dates hold for
any as_of — a "superseded" fee schedule is still the right source for a
question about last year. Retrieval (triage/answer.py) ranks every
version and only then resolves the winning chain by these dates, so the
quoted version is always the one in force at as_of.

Loading is strict: malformed front matter, dates with time components,
and dangling or asymmetric supersedes links all fail loudly here rather
than crashing (or silently mis-answering) at query time.
"""
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml

KB_DIR = Path(__file__).resolve().parents[1] / "kb"

_FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.S)
_REQUIRED = ("doc_id", "title", "effective_date")


@dataclass(frozen=True)
class KBDoc:
    doc_id: str
    title: str
    category: str
    version: int
    effective_date: date
    valid_until: date | None
    status: str
    supersedes: str | None
    superseded_by: str | None
    body: str

    def valid_at(self, as_of: date) -> bool:
        if as_of < self.effective_date:
            return False
        return self.valid_until is None or as_of <= self.valid_until

    @property
    def text(self) -> str:
        """What retrieval indexes: the title carries signal, so weight it in."""
        return f"{self.title}. {self.title}. {self.body}"


def parse_doc(raw: str, source: str) -> KBDoc:
    match = _FRONTMATTER.match(raw)
    if match is None:
        raise ValueError(f"{source}: missing YAML front matter")
    meta = yaml.safe_load(match.group(1)) or {}
    missing = [k for k in _REQUIRED if not meta.get(k)]
    if missing:
        raise ValueError(f"{source}: missing front-matter fields {missing}")
    # type() not isinstance(): datetime subclasses date, and a stray time
    # component would blow up date comparisons at query time instead.
    if type(meta["effective_date"]) is not date:
        raise ValueError(f"{source}: effective_date is not a plain date")
    valid_until = meta.get("valid_until") or None
    if valid_until is not None and type(valid_until) is not date:
        raise ValueError(f"{source}: valid_until is not a plain date")
    if valid_until is not None and valid_until < meta["effective_date"]:
        raise ValueError(f"{source}: valid_until precedes effective_date")
    return KBDoc(
        doc_id=str(meta["doc_id"]),
        title=str(meta["title"]),
        category=str(meta.get("category", "")),
        version=int(meta.get("version", 1)),
        effective_date=meta["effective_date"],
        valid_until=valid_until,
        status=str(meta.get("status", "")),
        supersedes=meta.get("supersedes") or None,
        superseded_by=meta.get("superseded_by") or None,
        body=re.sub(r"^#[^\n]*\n", "", match.group(2).strip()).strip(),
    )


def load_kb(kb_dir: str | Path = KB_DIR) -> list[KBDoc]:
    paths = sorted(Path(kb_dir).glob("*.md"))
    if not paths:
        raise ValueError(f"no KB documents found in {kb_dir}")
    # utf-8-sig: tolerate the BOM that Excel and some Windows editors add.
    docs = [parse_doc(p.read_text(encoding="utf-8-sig"), source=p.name) for p in paths]
    ids = [d.doc_id for d in docs]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate doc_ids in {kb_dir}")
    _check_links(docs)
    return docs


def _check_links(docs: list[KBDoc]) -> None:
    """Supersedes links must be symmetric — an asymmetric pair would either
    leave two versions 'in force' at once or strand a chain, so a broken
    link is a KB bug worth stopping the world for."""
    by_id = {d.doc_id: d for d in docs}
    for doc in docs:
        for field, reverse in (("superseded_by", "supersedes"),
                               ("supersedes", "superseded_by")):
            target_id = getattr(doc, field)
            if not target_id:
                continue
            target = by_id.get(target_id)
            if target is None:
                raise ValueError(
                    f"{doc.doc_id}: {field} points at unknown doc '{target_id}'"
                )
            if getattr(target, reverse) != doc.doc_id:
                raise ValueError(
                    f"{doc.doc_id} <-> {target_id}: "
                    f"{field}/{reverse} links disagree"
                )


def docs_valid_at(docs: list[KBDoc], as_of: date) -> list[KBDoc]:
    return [d for d in docs if d.valid_at(as_of)]
