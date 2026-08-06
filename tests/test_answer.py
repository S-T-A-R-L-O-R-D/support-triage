"""answer(question, as_of): grounding, temporal correctness, abstention.

The behaviours pinned here are the ones the exercise cares about:
the same question must cite different fee schedules for different as_of
dates, an ended promotion must be answered "no longer running" (not
silently skipped), and a question the KB cannot answer must be refused.
"""
from datetime import date

import pytest

from triage.answer import Answerer


@pytest.fixture(scope="session")
def answerer():
    return Answerer()


def test_current_fee_question_cites_current_schedule(answerer):
    result = answerer.answer(
        "What fee do I pay to withdraw crypto from my account?", date(2026, 7, 28)
    )
    assert result.doc_ids == ["kb-013"]
    assert "0.4%" in result.text


def test_same_question_on_historical_date_cites_superseded_schedule(answerer):
    result = answerer.answer(
        "What fee do I pay to withdraw crypto from my account?", date(2025, 6, 1)
    )
    assert result.doc_ids == ["kb-012"]
    assert "0.9%" in result.text


def test_ended_promotion_is_answered_as_ended_not_skipped(answerer):
    result = answerer.answer(
        "Is the invite a friend promotion still running?", date(2026, 7, 28)
    )
    assert result.doc_ids == ["kb-092"]
    assert "2026-03-31" in result.text


def test_question_outside_the_kb_is_refused(answerer):
    result = answerer.answer(
        "Do you offer margin trading with 10x leverage?", date(2026, 7, 28)
    )
    assert result.doc_ids == []
    assert not result.answered


def test_as_of_accepts_iso_string(answerer):
    result = answerer.answer("How long do I have to dispute a transaction?", "2026-03-01")
    assert result.doc_ids == ["kb-031"]  # the 60-day era, superseded since July


def test_empty_question_is_rejected(answerer):
    with pytest.raises(ValueError, match="question"):
        answerer.answer("   ", date(2026, 7, 28))


def test_unparseable_as_of_is_rejected(answerer):
    with pytest.raises(ValueError, match="as_of"):
        answerer.answer("What are the fees?", "28/07/2026")
