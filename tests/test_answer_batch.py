"""Batch answering CLI: output contract and row-level robustness."""
import csv

import pytest

from triage.answer_batch import main


def _write(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_cli_writes_qid_answer_doc_ids(tmp_path):
    inp, out = tmp_path / "q.csv", tmp_path / "a.csv"
    _write(
        inp,
        [
            {"qid": "x1", "question": "How long do I have to dispute a transaction?",
             "as_of": "2026-07-28"},
            {"qid": "x2", "question": "Do you offer margin trading with 10x leverage?",
             "as_of": "2026-07-28"},
        ],
        ["qid", "question", "as_of"],
    )

    main(["--input", str(inp), "--output", str(out)])

    rows = {r["qid"]: r for r in csv.DictReader(open(out, newline="", encoding="utf-8"))}
    assert set(rows) == {"x1", "x2"}
    assert set(rows["x1"]) == {"qid", "answer", "doc_ids"}
    assert rows["x1"]["doc_ids"] == "kb-032"  # 30-day window, current version
    assert rows["x2"]["doc_ids"] == ""  # refused: not in the KB
    assert rows["x2"]["answer"].startswith("Not answerable")


def test_row_with_bad_as_of_does_not_kill_the_batch(tmp_path):
    inp, out = tmp_path / "q.csv", tmp_path / "a.csv"
    _write(
        inp,
        [
            {"qid": "bad", "question": "What are the fees?", "as_of": "not-a-date"},
            {"qid": "good", "question": "How many price alerts can I have active at once?",
             "as_of": "2026-07-28"},
        ],
        ["qid", "question", "as_of"],
    )

    main(["--input", str(inp), "--output", str(out)])

    rows = {r["qid"]: r for r in csv.DictReader(open(out, newline="", encoding="utf-8"))}
    assert rows["good"]["doc_ids"] == "kb-105"
    assert rows["bad"]["doc_ids"] == ""


def test_cli_rejects_input_without_required_columns(tmp_path):
    inp = tmp_path / "q.csv"
    inp.write_text("id,text\n1,hello\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="question"):
        main(["--input", str(inp), "--output", str(tmp_path / "a.csv")])
