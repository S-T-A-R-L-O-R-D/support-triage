# Support Triage & KB Answering

Take-home practical: review of a handed-over route classifier (Part A), a
question-answering service over a versioned knowledge base (Part B), and
screenshot routing (Part C, light). No LLM anywhere — retrieval is TF-IDF,
answers are extractive, and that is a deliberate choice defended below.

## Setup & commands

```bash
uv sync --group dev          # or: pip install scikit-learn pyyaml pytest
uv run pytest                # 25 tests

# Entry point 1 — route classification (input: CSV with a `text` column)
uv run python -m triage.predict --input sample_messages.csv --output predictions.csv

# Entry point 2 — KB question answering
uv run python -m triage.answer_batch --input questions.csv --output answers.csv

# Honest baseline evaluation (Part A) and retrieval evaluation (Part B)
uv run python baseline/baseline_classifier.py
uv run python -m triage.eval_retrieval

# Part C — screenshot routing (runs from committed OCR cache;
# `uv sync --extra ocr` + `--refresh-ocr` to re-run real OCR)
uv run python -m triage.route_media --input media/screenshots --output media_routes.csv
```

Plain `python` works too if the dependencies are installed; run from the
repo root. `predictions.csv`, `answers.csv` and `media_routes.csv` in the
repo are the real outputs of the commands above.

---

## Part A — do I sign off? **No.**

**What's wrong with the evaluation.** The 98.75% is an artifact of data
leakage, twice over:

1. **Near-duplicate leakage.** The 400 training rows are templated: a core
   sentence wrapped in greeting/urgency boilerplate ("Urgent: X" /
   "Hey, X. Thanks."). Only **211 unique cores** exist; 262 rows are a
   variant of another row. Under the report's random 80/20 split, **49 of
   the 80 test rows had a near-twin in training** — the model was being
   asked questions it had effectively already seen.
2. **Vectorizer fitted before the split**, so test-set vocabulary and IDF
   statistics leaked into training.

Also: a single 80-row test set ("only one message wrong") is far too small
to support a ship decision, and plain accuracy on a 40%-majority-class
dataset says nothing about the route that actually matters.

**The honest number.** Grouping boilerplate variants so train and test
never share a core, fitting the vectorizer inside each fold
(5-fold stratified grouped CV — `baseline/baseline_classifier.py`, small
diff on their file):

| metric | reported | honest |
|---|---|---|
| accuracy | 98.75% | **88.5% ± 5.6** |
| macro F1 | — | **0.87** |
| fraud-report recall | — | **0.80** |

**The metric I'd hold it to in production.** Fraud-report is the most
expensive route to get wrong, and it's also the rarest (12.5%): so the
service metric is **fraud-report recall (floor ~0.95), monitored weekly**,
with macro F1 as overall health and per-route precision to watch queue
pollution. Today the model misroutes **10 of 50 fraud reports** (7 land in
account-access). Until recall is raised, I'd pair the model with a cheap
escalation rule (route to fraud on keyword/probability triggers even when
the argmax disagrees) and human review of low-confidence tickets — that
trade is far cheaper than a missed fraud case.

I tried `class_weight="balanced"`: fraud recall stayed at 0.80 while fraud
precision fell 0.85 → 0.76, so I kept the model unchanged — the evaluation
was the problem, not the model.

---

## Part B — answering from the KB

`answer(question, as_of)` → answer + documents used (`triage/answer.py`),
batch CLI above. Everything is deterministic and local.

**How the right version wins.** Documents are grouped into policy chains
via `supersedes`/`superseded_by`. Retrieval scores the question against
**every version** of every document (char 3–5-gram TF-IDF, cosine); the
winning chain is then resolved to the single version whose
`effective_date ≤ as_of ≤ valid_until`. Two properties fall out:

- an out-of-window version can never be quoted, by construction;
- a renamed policy still matches — "Withdrawal Fees" v1 text can win the
  similarity contest and still hand the answer to "Network and Transfer
  Charges" v3 (or v2 for a 2025 question), whichever is in force at `as_of`.

The `status` field is deliberately ignored for selection: it describes the
snapshot moment, while the dates hold for any `as_of` — a "superseded" fee
schedule is exactly the right source for "what was the fee in June 2025?".

**Ended notices are answers, not failures.** "Is the referral promotion
still running?" gets *"No — it applied from 2025-11-01 and ended on
2026-03-31"* citing the expired doc, rather than a silent refusal.

**Unanswerable questions.** If the best cosine score is below 0.22, the
system says *"Not answerable from the knowledge base"* with empty
`doc_ids`. Wrong-but-confident answers about money are worse than an
honest hand-off to a human; abstention keeps every emitted answer
traceable to a document that was in force on the date asked about.

**Extractive answers.** The winning doc's lead sentence (these policy docs
state the rule first) plus the sentence most similar to the question. The
system cannot say anything the KB does not contain. An LLM would phrase
answers more naturally, but adds a key, latency, cost, and a
hallucination/audit surface — for 31 documents the measurable step was
retrieval quality, so that's where the effort went.

**Honest evaluation** (`uv run python -m triage.eval_retrieval`; gold
labels hand-written per question in `eval/gold.csv`):

| metric | score |
|---|---|
| hit@1 on 31 answerable questions | **27/31 (87%)** |
| correct refusals on 7 out-of-KB questions | **6/7** |
| overall correct behaviour | **33/38 (87%)** |

Char n-grams were chosen over word n-grams by measurement: 28/31 vs 27/31
chain-level hit@1, and a better answer/refuse optimum (33/38 vs 31/38
overall) — they bridge "Dogecoin"→"DOGE". All five failures are
understood, not mysterious:

- **q19** "card purchase taking so long *right now*" → the card-fee doc
  outscores the live degradation notice (fee vocabulary dominates).
- **q22** "Dogecoin" scores 0.18, under the 0.22 threshold → false
  refusal (the top-ranked doc is the right one; the score is just weak).
- **q24** "new withdrawal address" → refused, and the refusal is the
  lesser evil: the fee docs (0.216) outrank the correct whitelist doc
  (0.203), so lowering the threshold would quote the wrong document.
  A ranking miss, not a threshold miss.
- **q32** "phone number for your fraud team" → retrieval matches the
  superseded 2FA doc (it mentions phone numbers twice) and date
  resolution hands the citation to its current successor, not the fraud
  doc. No phone number exists in the KB at all; the honest response is
  the in-app flow.
- **q38** "chargeback my own bank raised" shares dispute vocabulary with
  kb-032 → false answer where a refusal was due.

**Caveats stated plainly:** the abstention threshold was calibrated on
these same 38 questions (in-sample); the gold labels are my own reading of
the KB. On a held-back set I'd expect somewhat lower numbers. In
production this becomes: a held-out labelled set, refusal-rate and
score-distribution monitoring (a drift in either means the KB and the
questions have moved apart), and a "was this answer helpful" signal joined
back to the cited doc — a spike of negative feedback on one doc_id is the
stale-answer alarm.

---

## Part C — screenshots, light (stretch)

`uv run python -m triage.route_media ...` OCRs the three screenshots
locally and routes the text through the same classifier. All three route
correctly; the phishing SMS routes to fraud-report at 0.30 confidence and
is flagged `needs_review` — the degradation path doing its job.

- **Why screenshots over voice:** OCR of synthetic UI text is
  deterministic and cheap; speech adds accent/noise failure modes and a
  model download an order of magnitude larger, for the same demonstration.
- **Why local OCR (rapidocr/ONNX) over a vision API:** these images
  contain an email address, a phone number, balances and a wallet address.
  That data should not leave our infrastructure for a third-party OCR/vision
  API; on-infra OCR makes the privacy question moot. Cost: ~zero per
  ticket on CPU, ~0.5–2 s latency per image. A hosted vision model
  (~$0.002–0.01/image, 1–3 s) buys layout understanding we don't need here.
- **Degradation:** OCR lines under 0.8 confidence are dropped (kills
  status-bar garble); if too little text survives or route confidence is
  low, the ticket is flagged for a human instead of being trusted. A
  blurred screenshot degrades to "needs_review", not to a wrong queue.
- The OCR cache is committed (`media/ocr_cache/`), so the repo and tests
  run without the optional OCR dependency: `uv sync --extra ocr` and
  `--refresh-ocr` re-run the real engine.

---

## Scope & trade-offs

**Prioritized:** honest evaluation over model sophistication, in both
parts — the exercise is about knowing what your numbers mean. Temporal
correctness as a structural guarantee (filter by date, then it *cannot*
quote the wrong era) rather than a prompt instruction. Tests that pin the
behaviours that matter (32, all meaningful: leakage grouping, version
selection at historical dates, expired notices, abstention, validation,
KB link integrity, BOM tolerance, batch robustness, media routing).

**Deliberately left out:** any LLM (defensible without one, and the brief
says so); embedding models (char TF-IDF measured well enough on 31 docs;
embeddings are the first upgrade if the KB grows); multi-doc answers;
fixing q19/q32-style misses with special-casing (hand-tuned hacks that
wouldn't survive a held-back set); voice modality.

**With more time:** a held-out calibration set for the abstention
threshold; sentence-embedding retrieval behind the same
chain/date-resolution logic (q19 and q24 are ranking misses — semantic
similarity should separate "why is it slow" from "what does it cost",
which char n-grams cannot); probability
calibration + escalation rules for fraud recall; structured logging of
(question, score, cited doc, as_of) for the stale-answer monitoring
described above; CI running pytest + both evals.

**Time spent:** about 1.5 hours end-to-end. Built AI-assisted, as the
brief assumes — the commit history is correspondingly fast. My time went
into the scoping decisions (no LLM, which Part C modality, refusing to
special-case eval misses), reviewing every output and number before it
was committed, and this writeup; the decisions are mine to defend, and
the video walks through them.
