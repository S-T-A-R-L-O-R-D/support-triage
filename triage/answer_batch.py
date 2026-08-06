"""Batch question answering over the KB.

    python -m triage.answer_batch --input questions.csv --output answers.csv

Reads qid,question,as_of; writes qid,answer,doc_ids (semicolon-separated).
A malformed row is reported and skipped rather than killing the batch; a
missing as_of falls back to today, loudly.
"""
import argparse
import csv
import sys
from datetime import date

from triage.answer import Answerer


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV with qid,question[,as_of]")
    parser.add_argument("--output", required=True, help="where to write qid,answer,doc_ids")
    args = parser.parse_args(argv)

    try:
        with open(args.input, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fields = set(reader.fieldnames or [])
            if not {"qid", "question"} <= fields:
                raise ValueError(f"{args.input}: expected columns 'qid' and 'question'")
            rows = list(reader)
    except (OSError, ValueError) as exc:
        sys.exit(str(exc))

    answerer = Answerer()
    out_rows = []
    for row in rows:
        as_of = (row.get("as_of") or "").strip()
        if not as_of:
            as_of = date.today().isoformat()
            print(f"{row['qid']}: no as_of, using today ({as_of})", file=sys.stderr)
        try:
            result = answerer.answer(row["question"], as_of)
            answer_text, doc_ids = result.text, ";".join(result.doc_ids)
        except ValueError as exc:
            print(f"{row['qid']}: skipped ({exc})", file=sys.stderr)
            answer_text, doc_ids = f"Could not process this question: {exc}", ""
        out_rows.append({"qid": row["qid"], "answer": answer_text, "doc_ids": doc_ids})

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["qid", "answer", "doc_ids"])
        writer.writeheader()
        writer.writerows(out_rows)
    answered = sum(1 for r in out_rows if r["doc_ids"])
    print(f"wrote {len(out_rows)} answers ({answered} grounded, "
          f"{len(out_rows) - answered} refused/errored) to {args.output}")


if __name__ == "__main__":
    main()
