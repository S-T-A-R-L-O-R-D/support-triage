"""Batch route prediction.

    python -m triage.predict --input messages.csv --output predictions.csv

Reads a CSV with a `text` column, writes `text,prediction`.
"""
import argparse
import csv
import sys

from triage.classifier import DEFAULT_TRAIN_CSV, train


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV with a 'text' column")
    parser.add_argument("--output", required=True, help="where to write text,prediction")
    parser.add_argument("--train", default=DEFAULT_TRAIN_CSV, help="labelled training CSV")
    args = parser.parse_args(argv)

    try:
        with open(args.input, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or "text" not in reader.fieldnames:
                raise ValueError(f"{args.input}: expected a 'text' column")
            texts = [(r["text"] or "").strip() for r in reader]
        model = train(args.train)
    except (OSError, ValueError) as exc:
        sys.exit(str(exc))

    predictions = model.predict(texts) if texts else []
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "prediction"])
        writer.writeheader()
        for text, prediction in zip(texts, predictions):
            writer.writerow({"text": text, "prediction": prediction})
    print(f"wrote {len(texts)} predictions to {args.output}")


if __name__ == "__main__":
    main()
