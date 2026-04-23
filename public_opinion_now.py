from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline

from score_nuclear_articles import label_to_score


BEST_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
DEFAULT_QUERIES = [
    "nuclear energy",
    "nuclear power",
    "small modular reactor",
    "nuclear waste",
    "nuclear reactor",
]
HEADERS = {"User-Agent": "nuclear-sentiment-analysis/1.0 (+https://github.com/anikethbabu/nuclear_sentiment_analysis)"}


@dataclass
class PublicRunConfig:
    queries: list[str]
    model: str
    neutral_threshold: float
    max_items_per_source_query: int
    created_at_epoch: float


def clean_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_date(value: str | None) -> str:
    if not value:
        return ""
    try:
        return parsedate_to_datetime(value).isoformat()
    except Exception:
        return str(value)


def make_item_id(platform: str, url: str, title: str) -> str:
    key = url or f"{platform}:{title}"
    return f"{platform}:{abs(hash(key))}"


def fetch_google_news(query: str, limit: int) -> list[dict[str, Any]]:
    encoded = urllib.parse.quote_plus(f"{query} when:14d")
    url = f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"
    response = requests.get(url, headers=HEADERS, timeout=25)
    response.raise_for_status()
    root = ET.fromstring(response.content)
    rows = []
    for item in root.findall("./channel/item")[:limit]:
        title = clean_text(item.findtext("title"))
        link = clean_text(item.findtext("link"))
        summary = clean_text(item.findtext("description"))
        published = parse_date(item.findtext("pubDate"))
        source = clean_text(item.findtext("source")) or "Google News"
        rows.append(
            {
                "item_id": make_item_id("google_news", link, title),
                "platform": "google_news",
                "source": source,
                "query": query,
                "title": title,
                "text": clean_text(f"{title}. {summary}"),
                "url": link,
                "published_at": published,
                "engagement": np.nan,
            }
        )
    return rows


def fetch_reddit(query: str, limit: int) -> list[dict[str, Any]]:
    url = "https://www.reddit.com/search.json"
    params = {"q": f'"{query}"', "sort": "new", "t": "month", "limit": min(limit, 100), "raw_json": 1}
    response = requests.get(url, headers=HEADERS, params=params, timeout=25)
    response.raise_for_status()
    payload = response.json()
    rows = []
    for child in payload.get("data", {}).get("children", [])[:limit]:
        data = child.get("data", {})
        title = clean_text(data.get("title"))
        body = clean_text(data.get("selftext"))
        subreddit = clean_text(data.get("subreddit_name_prefixed") or data.get("subreddit"))
        permalink = data.get("permalink") or ""
        url = f"https://www.reddit.com{permalink}" if permalink.startswith("/") else clean_text(data.get("url"))
        created = data.get("created_utc")
        published = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(created)) if created else ""
        rows.append(
            {
                "item_id": make_item_id("reddit", url, title),
                "platform": "reddit",
                "source": subreddit or "reddit",
                "query": query,
                "title": title,
                "text": clean_text(f"{title}. {body}")[:5000],
                "url": url,
                "published_at": published,
                "engagement": float(data.get("score") or 0),
            }
        )
    return rows


def fetch_hacker_news(query: str, limit: int) -> list[dict[str, Any]]:
    url = "https://hn.algolia.com/api/v1/search_by_date"
    params = {"query": query, "tags": "story", "hitsPerPage": limit}
    response = requests.get(url, headers=HEADERS, params=params, timeout=25)
    response.raise_for_status()
    payload = response.json()
    rows = []
    for hit in payload.get("hits", [])[:limit]:
        title = clean_text(hit.get("title") or hit.get("story_title"))
        link = clean_text(hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID')}")
        rows.append(
            {
                "item_id": make_item_id("hacker_news", link, title),
                "platform": "hacker_news",
                "source": "Hacker News",
                "query": query,
                "title": title,
                "text": title,
                "url": link,
                "published_at": clean_text(hit.get("created_at")),
                "engagement": float(hit.get("points") or 0),
            }
        )
    return rows


def fetch_gdelt(query: str, limit: int) -> list[dict[str, Any]]:
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {"query": f'"{query}"', "mode": "ArtList", "format": "json", "maxrecords": limit, "sort": "HybridRel"}
    response = requests.get(url, headers=HEADERS, params=params, timeout=35)
    response.raise_for_status()
    payload = response.json()
    rows = []
    for article in payload.get("articles", [])[:limit]:
        title = clean_text(article.get("title"))
        link = clean_text(article.get("url"))
        source = clean_text(article.get("sourceCountry") or article.get("domain") or "GDELT")
        rows.append(
            {
                "item_id": make_item_id("gdelt", link, title),
                "platform": "gdelt",
                "source": source,
                "query": query,
                "title": title,
                "text": clean_text(f"{title}. {article.get('seendate', '')} {article.get('sourceCountry', '')}"),
                "url": link,
                "published_at": clean_text(article.get("seendate")),
                "engagement": np.nan,
            }
        )
    return rows


def score_to_public_label(score: float, threshold: float) -> str:
    if score > threshold:
        return "supportive"
    if score < -threshold:
        return "concerned"
    return "mixed_or_neutral"


def collect_public_items(queries: list[str], limit: int) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    fetchers = [
        ("google_news", fetch_google_news),
        ("reddit", fetch_reddit),
        ("hacker_news", fetch_hacker_news),
        ("gdelt", fetch_gdelt),
    ]
    rows = []
    errors = []
    for query in queries:
        for name, fetcher in fetchers:
            try:
                rows.extend(fetcher(query, limit))
            except Exception as exc:
                errors.append({"platform": name, "query": query, "error": f"{type(exc).__name__}: {exc}"})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame, errors
    frame = frame.drop_duplicates("item_id").reset_index(drop=True)
    frame = frame[frame["text"].str.len() > 5].copy()
    return frame, errors


def score_items(items: pd.DataFrame, model_name: str, threshold: float) -> pd.DataFrame:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    analyzer = pipeline("sentiment-analysis", model=model_name, tokenizer=tokenizer, truncation=True, max_length=512, device=-1)
    rows = []
    for row in tqdm(items.itertuples(index=False), total=len(items), desc="public opinion scoring"):
        result = analyzer(row.text[:4000])[0]
        tone_score = label_to_score(result["label"], result["score"])
        rows.append(
            {
                "item_id": row.item_id,
                "model": model_name,
                "tone_score": float(tone_score),
                "confidence": float(result["score"]),
                "predicted_public_stance": score_to_public_label(float(tone_score), threshold),
            }
        )
    return items.merge(pd.DataFrame(rows), on="item_id", how="left")


def summarize(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_platform = (
        scored.groupby("platform")
        .agg(
            items=("item_id", "count"),
            mean_tone_score=("tone_score", "mean"),
            median_tone_score=("tone_score", "median"),
            supportive_share=("predicted_public_stance", lambda x: (x == "supportive").mean()),
            concerned_share=("predicted_public_stance", lambda x: (x == "concerned").mean()),
            neutral_share=("predicted_public_stance", lambda x: (x == "mixed_or_neutral").mean()),
            mean_confidence=("confidence", "mean"),
        )
        .reset_index()
        .sort_values("items", ascending=False)
    )
    overall = pd.DataFrame(
        [
            {
                "platform": "ALL",
                "items": int(len(scored)),
                "mean_tone_score": float(scored["tone_score"].mean()),
                "median_tone_score": float(scored["tone_score"].median()),
                "supportive_share": float((scored["predicted_public_stance"] == "supportive").mean()),
                "concerned_share": float((scored["predicted_public_stance"] == "concerned").mean()),
                "neutral_share": float((scored["predicted_public_stance"] == "mixed_or_neutral").mean()),
                "mean_confidence": float(scored["confidence"].mean()),
            }
        ]
    )
    by_query = (
        scored.groupby("query")
        .agg(
            items=("item_id", "count"),
            mean_tone_score=("tone_score", "mean"),
            supportive_share=("predicted_public_stance", lambda x: (x == "supportive").mean()),
            concerned_share=("predicted_public_stance", lambda x: (x == "concerned").mean()),
        )
        .reset_index()
        .sort_values("mean_tone_score", ascending=False)
    )
    return pd.concat([overall, by_platform], ignore_index=True), by_query


def write_report(output_dir: Path, scored: pd.DataFrame, platform_summary: pd.DataFrame, query_summary: pd.DataFrame, errors: list[dict[str, str]]) -> None:
    overall = platform_summary[platform_summary["platform"] == "ALL"].iloc[0]
    lines = [
        "# Public Opinion Now",
        "",
        "## Interpretation",
        "",
        "This is a live directional sample from accessible public sources, not a statistically representative poll.",
        "It estimates current public/media stance from article titles/snippets and public discussion posts.",
        "",
        "## Headline",
        "",
        f"- Items scored: {int(overall['items'])}",
        f"- Mean tone score: {overall['mean_tone_score']:.3f}",
        f"- Predicted supportive share: {overall['supportive_share']:.1%}",
        f"- Predicted concerned share: {overall['concerned_share']:.1%}",
        f"- Predicted mixed/neutral share: {overall['neutral_share']:.1%}",
        "",
        "## By Platform",
        "",
        platform_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## By Topic Query",
        "",
        query_summary.to_markdown(index=False, floatfmt=".4f"),
    ]
    if errors:
        lines.extend(["", "## Collection Warnings", "", pd.DataFrame(errors).to_markdown(index=False)])
    lines.extend(
        [
            "",
            "## X/Twitter Note",
            "",
            "X/Twitter is not collected here because reliable search requires API access or credentials. Reddit and Hacker News are used as accessible public discussion proxies.",
        ]
    )
    (output_dir / "public_opinion_now_report.md").write_text("\n".join(lines), encoding="utf-8")


def save_to_sqlite(scored: pd.DataFrame, db_path: Path) -> None:
    con = sqlite3.connect(db_path)
    try:
        scored.to_sql("public_opinion_items", con, if_exists="replace", index=False)
    finally:
        con.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect current public nuclear-energy discussion and estimate public stance.")
    parser.add_argument("--output", default="models/public_now")
    parser.add_argument("--db", default="nuclear.db")
    parser.add_argument("--model", default=BEST_MODEL)
    parser.add_argument("--neutral-threshold", type=float, default=0.8)
    parser.add_argument("--max-items-per-source-query", type=int, default=25)
    parser.add_argument("--queries", nargs="*", default=DEFAULT_QUERIES)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = PublicRunConfig(
        queries=args.queries,
        model=args.model,
        neutral_threshold=args.neutral_threshold,
        max_items_per_source_query=args.max_items_per_source_query,
        created_at_epoch=time.time(),
    )
    (output_dir / "public_opinion_run_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    items, errors = collect_public_items(args.queries, args.max_items_per_source_query)
    if items.empty:
        raise SystemExit("No public items collected.")
    items.to_csv(output_dir / "public_items_raw.csv", index=False)
    scored = score_items(items, args.model, args.neutral_threshold)
    scored.to_csv(output_dir / "public_opinion_scored_items.csv", index=False)
    platform_summary, query_summary = summarize(scored)
    platform_summary.to_csv(output_dir / "public_opinion_by_platform.csv", index=False)
    query_summary.to_csv(output_dir / "public_opinion_by_query.csv", index=False)
    pd.DataFrame(errors).to_csv(output_dir / "public_collection_errors.csv", index=False)
    save_to_sqlite(scored, Path(args.db))
    write_report(output_dir, scored, platform_summary, query_summary, errors)
    print(platform_summary)
    print(f"report={output_dir / 'public_opinion_now_report.md'}")


if __name__ == "__main__":
    main()
