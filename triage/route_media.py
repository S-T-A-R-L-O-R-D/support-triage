"""Part C (light): route in-app screenshots into the four support routes.

    python -m triage.route_media --input media/screenshots --output media_routes.csv

OCR runs locally (rapidocr, ONNX) because these images hold exactly what a
customer would not want leaked — emails, balances, wallet addresses — so
they should never leave our infrastructure for a hosted OCR API. The
extracted text then goes through the same classifier as typed messages.

OCR output is cached in media/ocr_cache/ and the cache is committed, so
the repo (and the tests) run without the optional OCR dependency:
`uv sync --extra ocr` and `--refresh-ocr` re-run the real thing.

Degradation handling: OCR lines below 0.8 confidence are dropped (status
bars, garble); if too little text survives, or the classifier is unsure,
the row is flagged needs_review instead of being trusted.
"""
import argparse
import csv
import sys
from pathlib import Path

from triage.classifier import REPO_ROOT, train

CACHE_DIR = REPO_ROOT / "media" / "ocr_cache"
MIN_LINE_CONFIDENCE = 0.8
MIN_TEXT_CHARS = 40
MIN_ROUTE_CONFIDENCE = 0.5


def extract_text(image: Path, cache_dir: Path, refresh: bool = False) -> tuple[str, str]:
    """OCR an image, or reuse the committed cache. Returns (text, source)."""
    cache = cache_dir / (image.stem + ".txt")
    if cache.exists() and not refresh:
        return cache.read_text(encoding="utf-8"), "cache"
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        raise SystemExit(
            f"no cached OCR for {image.name} and rapidocr is not installed; "
            "run `uv sync --extra ocr` or restore media/ocr_cache/"
        )
    result, _ = RapidOCR()(str(image))
    lines = [text for _, text, conf in (result or []) if float(conf) >= MIN_LINE_CONFIDENCE]
    text = "\n".join(lines)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache.write_text(text, encoding="utf-8")
    return text, "ocr"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="directory of .png screenshots")
    parser.add_argument("--output", required=True, help="where to write routes CSV")
    parser.add_argument("--refresh-ocr", action="store_true", help="ignore the cache")
    args = parser.parse_args(argv)

    images = sorted(Path(args.input).glob("*.png"))
    if not images:
        sys.exit(f"no .png files in {args.input}")

    model = train()
    rows = []
    for image in images:
        text, source = extract_text(image, CACHE_DIR, refresh=args.refresh_ocr)
        proba = model.predict_proba([text])[0]
        route = model.classes_[proba.argmax()]
        confidence = float(proba.max())
        needs_review = len(text) < MIN_TEXT_CHARS or confidence < MIN_ROUTE_CONFIDENCE
        rows.append({
            "file": image.name,
            "route": route,
            "confidence": f"{confidence:.3f}",
            "source": source,
            "needs_review": "yes" if needs_review else "no",
        })

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["file", "route", "confidence", "source", "needs_review"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"routed {len(rows)} screenshots -> {args.output}")
    for r in rows:
        print(f"  {r['file']:22s} {r['route']:20s} conf={r['confidence']} "
              f"review={r['needs_review']} ({r['source']})")


if __name__ == "__main__":
    main()
