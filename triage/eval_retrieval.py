"""Honest retrieval evaluation against hand-labelled gold.

    python -m triage.eval_retrieval

For every question in questions.csv, compares the doc the answerer cites
against eval/gold.csv and reports hit@1 on answerable questions, refusal
precision/recall on out-of-KB questions, and every failure by type.

Caveat, stated where the numbers are printed: the abstention threshold
was calibrated on these same 38 questions. Treat the numbers as
in-sample; a held-back set (like the graders') is the real test.
"""
import csv
from pathlib import Path

from triage.answer import Answerer
from triage.classifier import REPO_ROOT

QUESTIONS = REPO_ROOT / "questions.csv"
GOLD = Path(__file__).resolve().parents[1] / "eval" / "gold.csv"


def main() -> None:
    with open(QUESTIONS, newline="", encoding="utf-8") as f:
        questions = list(csv.DictReader(f))
    with open(GOLD, newline="", encoding="utf-8") as f:
        gold = {r["qid"]: r["expected_doc"] for r in csv.DictReader(f)}
    missing = [q["qid"] for q in questions if q["qid"] not in gold]
    if missing:
        raise SystemExit(f"no gold label for {missing}")

    answerer = Answerer()
    failures = []
    hits = n_answerable = correct_refusals = n_unanswerable = false_answers = 0
    for q in questions:
        result = answerer.answer(q["question"], q["as_of"])
        cited = result.doc_ids[0] if result.doc_ids else "NONE"
        expected = gold[q["qid"]]
        if expected == "NONE":
            n_unanswerable += 1
            if cited == "NONE":
                correct_refusals += 1
            else:
                false_answers += 1
                failures.append((q["qid"], "false-answer",
                                 f"cited {cited}, should refuse (score {result.score:.2f})"))
        else:
            n_answerable += 1
            if cited == expected:
                hits += 1
            elif cited == "NONE":
                failures.append((q["qid"], "false-refusal",
                                 f"should cite {expected} (score {result.score:.2f})"))
            else:
                failures.append((q["qid"], "wrong-doc",
                                 f"cited {cited}, expected {expected}"))

    total_ok = hits + correct_refusals
    print(f"answerable questions   hit@1: {hits}/{n_answerable}")
    print(f"out-of-KB questions  refused: {correct_refusals}/{n_unanswerable}")
    print(f"overall correct behaviour:   {total_ok}/{len(questions)}"
          f" ({total_ok / len(questions):.0%})")
    print("\nfailures:")
    for qid, kind, detail in failures:
        print(f"  {qid:4s} {kind:13s} {detail}")
    print("\ncaveat: threshold calibrated on these same questions (in-sample).")


if __name__ == "__main__":
    main()
