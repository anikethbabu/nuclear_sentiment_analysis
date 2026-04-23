from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline


DEFAULT_MODELS = [
    "distilbert-base-uncased-finetuned-sst-2-english",
    "ProsusAI/finbert",
    "cardiffnlp/twitter-roberta-base-sentiment-latest",
]


@dataclass
class ScoreConfig:
    db_path: str
    table: str
    output_dir: str
    max_articles: int | None
    max_chunks_per_article: int
    models: list[str]
    created_at_epoch: float


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_articles(db_path: str, table: str, max_articles: int | None) -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table}", con)
    finally:
        con.close()

    required = {"id", "source", "label", "title", "url", "content"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {table}: {missing}")

    df = df.copy()
    df = df.rename(columns={"label": "source_type"})
    df["title"] = df["title"].map(normalize_text)
    df["content"] = df["content"].map(normalize_text)
    df["text"] = (df["title"] + ". " + df["content"]).map(normalize_text)
    df = df[df["text"].str.len() > 20].drop_duplicates(subset=["url"]).reset_index(drop=True)
    df["source_type"] = df["source_type"].fillna("unknown").map(lambda x: str(x).strip().lower())
    if max_articles:
        df = df.head(max_articles).copy()
    return df


def label_to_score(label: str, score: float) -> float:
    normalized = label.lower().strip()
    if normalized in {"positive", "label_2"}:
        return float(score)
    if normalized in {"negative", "label_0"}:
        return -float(score)
    return 0.0


def chunk_text(text: str, tokenizer: Any, max_chunks: int) -> list[str]:
    encoded = tokenizer(
        text,
        truncation=True,
        return_overflowing_tokens=True,
        max_length=min(getattr(tokenizer, "model_max_length", 512), 512),
        stride=32,
    )
    chunks = [tokenizer.decode(ids, skip_special_tokens=True) for ids in encoded["input_ids"]]
    return chunks[:max_chunks]


def score_with_model(df: pd.DataFrame, model_name: str, max_chunks: int) -> pd.DataFrame:
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
    for row in tqdm(df.itertuples(index=False), total=len(df), desc=model_name):
        chunks = chunk_text(row.text, tokenizer, max_chunks)
        chunk_scores = []
        chunk_confidences = []
        for chunk in chunks:
            result = analyzer(chunk)[0]
            chunk_scores.append(label_to_score(result["label"], result["score"]))
            chunk_confidences.append(float(result["score"]))
        rows.append(
            {
                "id": row.id,
                "model": model_name,
                "tone_score": float(np.mean(chunk_scores)) if chunk_scores else 0.0,
                "tone_score_std": float(np.std(chunk_scores)) if len(chunk_scores) > 1 else 0.0,
                "model_confidence_mean": float(np.mean(chunk_confidences)) if chunk_confidences else 0.0,
                "chunks_scored": len(chunks),
            }
        )
    return pd.DataFrame(rows)


def summarize_scores(article_scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_cols = [col for col in article_scores.columns if col.startswith("tone_score_") and not col.endswith("_std")]
    article_scores["ensemble_tone_score"] = article_scores[score_cols].mean(axis=1)
    article_scores["model_disagreement"] = article_scores[score_cols].std(axis=1).fillna(0.0)

    summary = (
        article_scores.groupby(["source"], dropna=False)
        .agg(
            articles=("id", "count"),
            mean_tone_score=("ensemble_tone_score", "mean"),
            median_tone_score=("ensemble_tone_score", "median"),
            tone_score_std=("ensemble_tone_score", "std"),
            q10_tone_score=("ensemble_tone_score", lambda x: x.quantile(0.10)),
            q90_tone_score=("ensemble_tone_score", lambda x: x.quantile(0.90)),
            mean_model_disagreement=("model_disagreement", "mean"),
        )
        .reset_index()
        .sort_values("mean_tone_score", ascending=False)
    )

    overall = pd.DataFrame(
        [
            {
                "articles": int(len(article_scores)),
                "mean_tone_score": float(article_scores["ensemble_tone_score"].mean()),
                "median_tone_score": float(article_scores["ensemble_tone_score"].median()),
                "tone_score_std": float(article_scores["ensemble_tone_score"].std()),
                "q10_tone_score": float(article_scores["ensemble_tone_score"].quantile(0.10)),
                "q90_tone_score": float(article_scores["ensemble_tone_score"].quantile(0.90)),
                "mean_model_disagreement": float(article_scores["model_disagreement"].mean()),
            }
        ]
    )
    return article_scores, pd.concat([overall.assign(source="ALL"), summary], ignore_index=True)


def write_report(output_dir: Path, df: pd.DataFrame, summary: pd.DataFrame, models: list[str]) -> None:
    lines = [
        "# Nuclear Article Tone Report",
        "",
        "## Method",
        "",
        "The SQLite `label` column is treated only as `source_type`; it is not used as model ground truth.",
        "No supervised sentiment model is trained from that column. Scores are model-estimated continuous tone values in the range -1 to 1.",
        "",
        "## Data",
        "",
        f"- Articles scored: {len(df)}",
        f"- Models used: `{models}`",
        f"- Source counts: `{df['source'].value_counts().to_dict()}`",
        "",
        "## Summary Statistics",
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Notes",
        "",
        "The SQLite source-type metadata is preserved in per-article output, but it is not used for scoring or summary statistics. Use the continuous `ensemble_tone_score` and `model_disagreement` columns for analysis.",
    ]
    (output_dir / "tone_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score nuclear articles without using SQLite source-type labels as sentiment.")
    parser.add_argument("--db", default="nuclear.db")
    parser.add_argument("--table", default="articles")
    parser.add_argument("--output", default="models/tone")
    parser.add_argument("--max-articles", type=int, default=None)
    parser.add_argument("--max-chunks-per-article", type=int, default=8)
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = ScoreConfig(
        db_path=args.db,
        table=args.table,
        output_dir=args.output,
        max_articles=args.max_articles,
        max_chunks_per_article=args.max_chunks_per_article,
        models=args.models,
        created_at_epoch=time.time(),
    )
    (output_dir / "tone_run_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    df = load_articles(args.db, args.table, args.max_articles)
    df[["id", "source", "source_type", "title", "url", "text"]].to_csv(output_dir / "scored_article_input.csv", index=False)

    model_frames = [score_with_model(df, model, args.max_chunks_per_article) for model in args.models]
    long_scores = pd.concat(model_frames, ignore_index=True)
    long_scores.to_csv(output_dir / "model_tone_scores_long.csv", index=False)

    wide = df[["id", "source", "source_type", "title", "url"]].copy()
    for frame in model_frames:
        model_name = frame["model"].iloc[0].replace("/", "__")
        renamed = frame.rename(
            columns={
                "tone_score": f"tone_score_{model_name}",
                "tone_score_std": f"tone_score_std_{model_name}",
                "model_confidence_mean": f"confidence_{model_name}",
                "chunks_scored": f"chunks_{model_name}",
            }
        ).drop(columns=["model"])
        wide = wide.merge(renamed, on="id", how="left")

    article_scores, summary = summarize_scores(wide)
    article_scores.to_csv(output_dir / "article_tone_scores.csv", index=False)
    summary.to_csv(output_dir / "tone_summary_by_source_type.csv", index=False)
    write_report(output_dir, df, summary, args.models)
    print(f"report={output_dir / 'tone_report.md'}")


if __name__ == "__main__":
    main()
