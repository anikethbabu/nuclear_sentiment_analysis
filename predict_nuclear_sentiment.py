from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict nuclear article sentiment/stance with the trained model.")
    parser.add_argument("--model", default="models/best_nuclear_sentiment_model.joblib")
    parser.add_argument("--text", help="Text to classify.")
    parser.add_argument("--file", help="Plain-text file to classify.")
    args = parser.parse_args()

    if not args.text and not args.file:
        raise SystemExit("Provide --text or --file.")

    text = args.text if args.text else Path(args.file).read_text(encoding="utf-8")
    model = joblib.load(args.model)
    pred = model.predict([text])[0]
    print(f"prediction: {pred}")
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([text])[0]
        rows = sorted(zip(model.classes_, probs), key=lambda x: x[1], reverse=True)
        print(pd.DataFrame(rows, columns=["label", "probability"]).to_string(index=False))


if __name__ == "__main__":
    main()
