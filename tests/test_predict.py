"""Route classifier + predict CLI: behaviour, output contract, validation."""
import csv

import pytest

from triage.classifier import DEFAULT_TRAIN_CSV, ROUTES, train
from triage.predict import main


@pytest.fixture(scope="session")
def model():
    return train(DEFAULT_TRAIN_CSV)


def test_routes_an_obvious_fraud_report(model):
    text = "Someone withdrew $500 of BTC from my account that I never authorized."
    assert model.predict([text])[0] == "fraud-report"


def test_routes_an_obvious_access_problem(model):
    text = "I can't log into my account, the password reset email never arrives."
    assert model.predict([text])[0] == "account-access"


def test_cli_writes_text_and_prediction_columns(tmp_path):
    inp = tmp_path / "messages.csv"
    out = tmp_path / "predictions.csv"
    with open(inp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["text"])
        w.writeheader()
        w.writerow({"text": "I was charged twice for the same purchase."})
        w.writerow({"text": "How do I enable price alerts?"})

    main(["--input", str(inp), "--output", str(out)])

    with open(out, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [set(r) for r in rows] == [{"text", "prediction"}] * 2
    assert all(r["prediction"] in ROUTES for r in rows)


def test_cli_rejects_input_without_text_column(tmp_path):
    inp = tmp_path / "messages.csv"
    inp.write_text("message\nhello\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="text"):
        main(["--input", str(inp), "--output", str(tmp_path / "out.csv")])
