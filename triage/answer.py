"""Grounded question answering over the KB.

Retrieval strategy: score the question against *every* document version,
then resolve which version of the winning policy chain to trust purely by
`as_of` dates. Ranking on content across versions means renamed policies
("Withdrawal Fees" -> "Network and Transfer Charges") still match however
the customer phrases it; the date resolution guarantees an out-of-window
version can never be quoted.

Answers are extractive — the most relevant sentences of the winning
document — so they cannot say anything the KB does not. No LLM involved.

Character n-grams (3-5) beat word n-grams on this corpus (hit@1 29/31 vs
28/31, and a wider answerable/unanswerable score gap) because they match
across surface variation: "Dogecoin" ~ "DOGE", "maintenance" ~
"maintenance window". Config chosen by measurement, see eval/.

Three outcomes:
  * answered from the version in force at as_of;
  * "no longer in force" when the best match is a dated notice/promotion
    that ended before as_of (that is an answer, not a failure);
  * refusal when the best match is too weak (score < min_score) or no
    version existed yet at as_of.
"""
import re
from dataclasses import dataclass
from datetime import date
from functools import lru_cache

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from triage.kb import KB_DIR, KBDoc, load_kb

# Below this cosine similarity the best match is noise, not an answer.
# Calibrated on the 38 labelled questions: out-of-KB questions score
# <= 0.20 (one outlier at 0.29), answerable ones >= 0.23 (one at 0.19).
# Known trade-off documented in the README; recalibrate when the KB moves.
MIN_SCORE = 0.22

REFUSAL = "Not answerable from the knowledge base as of {as_of}."

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Answer:
    text: str
    doc_ids: list[str]
    score: float
    answered: bool


class Answerer:
    def __init__(self, kb_dir=KB_DIR, min_score: float = MIN_SCORE):
        self.docs = load_kb(kb_dir)
        self.min_score = min_score
        self._vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True
        )
        self._doc_matrix = self._vectorizer.fit_transform([d.text for d in self.docs])
        self._chains = self._build_chains()

    def _build_chains(self) -> dict[str, list[KBDoc]]:
        """Group versions of one policy; key every member by the chain head."""
        by_id = {d.doc_id: d for d in self.docs}
        chains: dict[str, list[KBDoc]] = {}
        heads: dict[str, str] = {}
        for doc in self.docs:
            current, seen = doc, {doc.doc_id}
            while current.superseded_by:
                current = by_id[current.superseded_by]
                if current.doc_id in seen:
                    raise ValueError(f"supersedes cycle at {doc.doc_id}")
                seen.add(current.doc_id)
            heads[doc.doc_id] = current.doc_id
            chains.setdefault(current.doc_id, []).append(doc)
        return {d.doc_id: sorted(chains[heads[d.doc_id]], key=lambda x: x.effective_date)
                for d in self.docs}

    def answer(self, question: str, as_of: date | str) -> Answer:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        as_of = _parse_as_of(as_of)

        scores = cosine_similarity(
            self._vectorizer.transform([question]), self._doc_matrix
        )[0]
        best = self.docs[int(np.argmax(scores))]
        score = float(scores.max())
        if score < self.min_score:
            return Answer(REFUSAL.format(as_of=as_of), [], score, answered=False)

        chain = self._chains[best.doc_id]
        in_force = [d for d in chain if d.valid_at(as_of)]
        if in_force:
            doc = max(in_force, key=lambda d: d.effective_date)
            return Answer(self._extract(question, doc), [doc.doc_id], score, answered=True)

        ended = [d for d in chain if d.valid_until and d.valid_until < as_of]
        if ended:
            doc = max(ended, key=lambda d: d.valid_until)
            text = (
                f'No — "{doc.title}" is no longer in force: it applied from '
                f"{doc.effective_date} and ended on {doc.valid_until}."
            )
            return Answer(text, [doc.doc_id], score, answered=True)

        # The policy only exists in versions that start after as_of.
        return Answer(REFUSAL.format(as_of=as_of), [], score, answered=False)

    def _extract(self, question: str, doc: KBDoc) -> str:
        """The 1-2 sentences of the winning doc most similar to the question."""
        sentences = _SENTENCE_SPLIT.split(doc.body.replace("\n", " "))
        if len(sentences) <= 2:
            return " ".join(sentences)
        similarity = cosine_similarity(
            self._vectorizer.transform([question]),
            self._vectorizer.transform(sentences),
        )[0]
        top = sorted(np.argsort(similarity)[-2:])  # best two, document order
        return " ".join(sentences[i] for i in top)


def _parse_as_of(as_of: date | str) -> date:
    if isinstance(as_of, date):
        return as_of
    try:
        return date.fromisoformat(str(as_of).strip())
    except ValueError:
        raise ValueError(f"as_of must be an ISO date (YYYY-MM-DD), got {as_of!r}") from None


@lru_cache(maxsize=1)
def _default_answerer() -> Answerer:
    return Answerer()


def answer(question: str, as_of: date | str) -> Answer:
    """Module-level convenience: answer(question, as_of) -> Answer."""
    return _default_answerer().answer(question, as_of)
