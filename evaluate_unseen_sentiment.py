from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score

from score_nuclear_articles import DEFAULT_MODELS, chunk_text, label_to_score


ARTICLE_FOLDERS = {
    "ANS": Path("ans_articles"),
    "World Nuclear": Path("World_Nuclear_Scraper") / "articles",
}
LABEL_ORDER = ["Negative", "Neutral", "Positive"]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\ufeff", " ")).strip()


def normalize_key(value: str) -> str:
    fixed = value.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")
    return re.sub(r"\s+", " ", fixed or value).strip()


def load_external_articles(external_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source, relative_folder in ARTICLE_FOLDERS.items():
        folder = external_root / relative_folder
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.txt")):
            text = normalize_text(path.read_text(encoding="utf-8", errors="replace"))
            if not text:
                continue
            rows.append(
                {
                    "article_id": f"{source}::{path.name}",
                    "article_key": normalize_key(f"{source}::{path.name}"),
                    "source": source,
                    "filename": path.name,
                    "text": text,
                    "word_count": len(text.split()),
                }
            )
    return pd.DataFrame(rows).drop_duplicates("article_key").reset_index(drop=True)


def load_labeled_unseen(external_root: Path, labels_path: Path) -> pd.DataFrame:
    articles = load_external_articles(external_root)
    labels = pd.read_csv(labels_path)
    labels["article_key"] = labels["article_id"].map(normalize_key)
    labels["label"] = labels["label"].str.strip().str.title()
    labeled = articles.merge(labels[["article_key", "label"]], on="article_key", how="inner")
    if labeled.empty:
        raise ValueError("No article files matched the provided labels.")
    return labeled


def score_model(df: pd.DataFrame, model_name: str, max_chunks: int) -> pd.DataFrame:
    from tqdm import tqdm
    from transformers import AutoTokenizer, pipeline

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    analyzer = pipeline(
        "sentiment-analysis",
        model=model_name,
        tokenizer=tokenizer,
        truncation=True,
        max_length=512,
        device=-1,
    )
    rows: list[dict[str, Any]] = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc=f"unseen {model_name}"):
        chunks = chunk_text(row.text, tokenizer, max_chunks)
        scores = []
        confidences = []
        for chunk in chunks:
            result = analyzer(chunk)[0]
            scores.append(label_to_score(result["label"], result["score"]))
            confidences.append(float(result["score"]))
        rows.append(
            {
                "article_id": row.article_id,
                "model": model_name,
                "tone_score": float(np.mean(scores)) if scores else 0.0,
                "confidence_mean": float(np.mean(confidences)) if confidences else 0.0,
                "chunks_scored": len(chunks),
            }
        )
    return pd.DataFrame(rows)


def score_to_label(score: float, neutral_threshold: float) -> str:
    if score > neutral_threshold:
        return "Positive"
    if score < -neutral_threshold:
        return "Negative"
    return "Neutral"


def metric_block(y_true: pd.Series, y_pred: pd.Series) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "classification_report": classification_report(
            y_true, y_pred, labels=LABEL_ORDER, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABEL_ORDER).tolist(),
    }


def evaluate_thresholds(scores: pd.DataFrame, truth: pd.DataFrame, thresholds: list[float]) -> tuple[pd.DataFrame, dict[str, Any]]:
    merged = truth[["article_id", "label"]].merge(scores, on="article_id", how="inner")
    rows = []
    details: dict[str, Any] = {}
    for model_name, frame in merged.groupby("model"):
        for threshold in thresholds:
            pred = frame["tone_score"].map(lambda value: score_to_label(value, threshold))
            metrics = metric_block(frame["label"], pred)
            rows.append(
                {
                    "model": model_name,
                    "neutral_threshold": threshold,
                    "accuracy": metrics["accuracy"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "weighted_f1": metrics["weighted_f1"],
                    "matched_rows": int(len(frame)),
                }
            )
            details[f"{model_name}|threshold={threshold}"] = metrics

    wide = merged.pivot_table(index="article_id", columns="model", values="tone_score", aggfunc="first").reset_index()
    wide["tone_score"] = wide.drop(columns=["article_id"]).mean(axis=1)
    ensemble = truth[["article_id", "label"]].merge(wide[["article_id", "tone_score"]], on="article_id", how="inner")
    for threshold in thresholds:
        pred = ensemble["tone_score"].map(lambda value: score_to_label(value, threshold))
        metrics = metric_block(ensemble["label"], pred)
        rows.append(
            {
                "model": "ensemble_mean",
                "neutral_threshold": threshold,
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "matched_rows": int(len(ensemble)),
            }
        )
        details[f"ensemble_mean|threshold={threshold}"] = metrics

    ranking = pd.DataFrame(rows).sort_values(["macro_f1", "balanced_accuracy", "accuracy"], ascending=False)
    return ranking, details


def write_report(output_dir: Path, labeled: pd.DataFrame, ranking: pd.DataFrame, details: dict[str, Any]) -> None:
    best = ranking.iloc[0]
    best_key = f"{best['model']}|threshold={best['neutral_threshold']}"
    lines = [
        "# Unseen Accuracy Report",
        "",
        "## Benchmark",
        "",
        "This evaluates pretrained sentiment scorers on the unseen labeled articles from `UnknowStudio4/labels.csv`.",
        "No model is trained on the benchmark labels in this script.",
        "",
        f"- Matched labeled articles: {len(labeled)}",
        f"- Label counts: `{labeled['label'].value_counts().to_dict()}`",
        "",
        "## Ranking",
        "",
        ranking.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Best Run",
        "",
        f"- Model: `{best['model']}`",
        f"- Neutral threshold: `{best['neutral_threshold']}`",
        f"- Accuracy: `{best['accuracy']:.4f}`",
        f"- Balanced accuracy: `{best['balanced_accuracy']:.4f}`",
        f"- Macro F1: `{best['macro_f1']:.4f}`",
        "",
        "Confusion matrix label order: `Negative`, `Neutral`, `Positive`",
        "",
        "```json",
        json.dumps(details[best_key]["confusion_matrix"], indent=2),
        "```",
        "",
        "Classification report:",
        "",
        "```json",
        json.dumps(details[best_key]["classification_report"], indent=2),
        "```",
    ]
    (output_dir / "unseen_accuracy_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate sentiment scorers on unseen labeled external articles.")
    parser.add_argument("--external-root", default=str(Path("..") / "UnknowStudio4"))
    parser.add_argument("--labels", default="")
    parser.add_argument("--output", default="models/unseen_eval")
    parser.add_argument("--max-chunks-per-article", type=int, default=8)
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--thresholds", nargs="*", type=float, default=[0.05, 0.10, 0.20, 0.30])
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    external_root = Path(args.external_root)
    labels_path = Path(args.labels) if args.labels else external_root / "labels.csv"

    labeled = load_labeled_unseen(external_root, labels_path)
    labeled[["article_id", "source", "filename", "label", "word_count"]].to_csv(
        output_dir / "unseen_labeled_articles.csv", index=False
    )
    score_frames = [score_model(labeled, model, args.max_chunks_per_article) for model in args.models]
    scores = pd.concat(score_frames, ignore_index=True)
    scores.to_csv(output_dir / "unseen_model_scores_long.csv", index=False)

    ranking, details = evaluate_thresholds(scores, labeled, args.thresholds)
    ranking.to_csv(output_dir / "unseen_accuracy_ranking.csv", index=False)
    (output_dir / "unseen_accuracy_details.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    write_report(output_dir, labeled, ranking, details)
    print(ranking.head(10))
    print(f"report={output_dir / 'unseen_accuracy_report.md'}")


if __name__ == "__main__":
    main()
