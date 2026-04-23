from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


ROOT = Path(__file__).parent
DB_PATH = ROOT / "nuclear.db"
TONE_DIR = ROOT / "models" / "tone"
UNSEEN_DIR = ROOT / "models" / "unseen_eval"


st.set_page_config(page_title="Nuclear Article Tone Brief", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem; padding-bottom: 1.2rem; max-width: 1440px;}
    h1 {font-size: 2.25rem; letter-spacing: 0; margin-bottom: 0.2rem;}
    h2, h3 {letter-spacing: 0;}
    [data-testid="stMetric"] {
        border: 1px solid #d8dee8;
        border-radius: 8px;
        padding: 0.8rem 0.9rem;
        background: #ffffff;
        box-shadow: 0 1px 2px rgba(20, 32, 55, 0.04);
    }
    .takeaway {
        border-left: 4px solid #2374ab;
        padding: 0.75rem 1rem;
        background: #f5f8fb;
        border-radius: 6px;
        margin: 0.35rem 0 0.7rem 0;
    }
    .small-note {color: #5c6675; font-size: 0.92rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_db_counts() -> dict[str, int]:
    con = sqlite3.connect(DB_PATH)
    try:
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('articles', 'external_articles')",
            con,
        )["name"].tolist()
        counts = {}
        for table in tables:
            counts[table] = int(pd.read_sql_query(f"SELECT COUNT(*) AS n FROM {table}", con)["n"].iloc[0])
        return counts
    finally:
        con.close()


@st.cache_data(show_spinner=False)
def load_sources() -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)
    try:
        return pd.read_sql_query("SELECT source, COUNT(*) AS articles FROM articles GROUP BY source", con)
    finally:
        con.close()


@st.cache_data(show_spinner=False)
def load_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    article_scores = pd.read_csv(TONE_DIR / "article_tone_scores.csv")
    source_summary = pd.read_csv(TONE_DIR / "tone_summary_by_source.csv")
    ranking = pd.read_csv(UNSEEN_DIR / "unseen_accuracy_ranking.csv")
    details = json.loads((UNSEEN_DIR / "unseen_accuracy_details.json").read_text(encoding="utf-8"))
    return article_scores, source_summary, ranking, details


counts = load_db_counts()
source_counts = load_sources()
article_scores, source_summary, unseen_ranking, unseen_details = load_outputs()

best = unseen_ranking.iloc[0]
best_key = f"{best['model']}|threshold={best['neutral_threshold']}"
best_details = unseen_details[best_key]
overall = source_summary[source_summary["source"] == "ALL"].iloc[0]
by_source = source_summary[source_summary["source"] != "ALL"].copy()

st.title("Nuclear Article Tone Brief")
st.markdown(
    "<div class='small-note'>SQLite source-rating columns were removed. Tone uses pretrained scorers; accuracy uses only external ground-truth labels.</div>",
    unsafe_allow_html=True,
)

metric_cols = st.columns([1, 1, 1, 1, 1])
metric_cols[0].metric("Original Articles", f"{counts.get('articles', 0):,}")
metric_cols[1].metric("Imported Articles", f"{counts.get('external_articles', 0):,}")
metric_cols[2].metric("Mean Tone", f"{overall['mean_tone_score']:.3f}")
metric_cols[3].metric("Best Accuracy", f"{best['accuracy']:.0%}")
metric_cols[4].metric("Best Macro F1", f"{best['macro_f1']:.3f}")

st.markdown(
    f"""
    <div class='takeaway'>
    <b>Presenter takeaway:</b> the corrected pipeline separates source metadata from sentiment. On the external labeled benchmark,
    <b>{best['model']}</b> performs best with <b>{best['accuracy']:.0%}</b> accuracy and <b>{best['macro_f1']:.3f}</b> macro F1.
    The original corpus is nearly tone-balanced overall, with source-level differences visible below.
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1.15, 0.85])

tone_chart = (
    alt.Chart(by_source)
    .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
    .encode(
        y=alt.Y("source:N", sort="-x", title="Source"),
        x=alt.X("mean_tone_score:Q", title="Mean ensemble tone score", scale=alt.Scale(domain=[-0.15, 0.25])),
        color=alt.Color(
            "mean_tone_score:Q",
            scale=alt.Scale(domain=[-0.12, 0, 0.22], range=["#b64242", "#8b98a8", "#2374ab"]),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("source:N", title="Source"),
            alt.Tooltip("articles:Q", title="Articles"),
            alt.Tooltip("mean_tone_score:Q", title="Mean tone", format=".3f"),
            alt.Tooltip("mean_model_disagreement:Q", title="Model disagreement", format=".3f"),
        ],
    )
    .properties(height=310)
)

with left:
    st.subheader("Tone by Source")
    st.altair_chart(tone_chart, use_container_width=True)

with right:
    st.subheader("Article Mix")
    source_mix = (
        alt.Chart(source_counts)
        .mark_arc(innerRadius=58, outerRadius=108)
        .encode(
            theta=alt.Theta("articles:Q"),
            color=alt.Color(
                "source:N",
                scale=alt.Scale(range=["#2374ab", "#f28e2b", "#59a14f", "#e15759", "#7f6aad"]),
                legend=alt.Legend(title=None, orient="bottom"),
            ),
            tooltip=["source:N", "articles:Q"],
        )
        .properties(height=310)
    )
    st.altair_chart(source_mix, use_container_width=True)

st.subheader("Unseen Ground-Truth Benchmark")
rank_view = unseen_ranking.copy()
rank_view["run"] = rank_view["model"].str.replace("distilbert-base-uncased-finetuned-sst-2-english", "DistilBERT SST-2", regex=False)
rank_view["run"] = rank_view["run"].str.replace("cardiffnlp/twitter-roberta-base-sentiment-latest", "Cardiff RoBERTa", regex=False)
rank_view["run"] = rank_view["run"].str.replace("ProsusAI/finbert", "FinBERT", regex=False)

bench_chart = (
    alt.Chart(rank_view)
    .mark_circle(size=130, opacity=0.85)
    .encode(
        x=alt.X("accuracy:Q", title="Accuracy", scale=alt.Scale(domain=[0.2, 0.9])),
        y=alt.Y("macro_f1:Q", title="Macro F1", scale=alt.Scale(domain=[0.2, 0.9])),
        color=alt.Color(
            "run:N",
            scale=alt.Scale(range=["#2374ab", "#e15759", "#59a14f", "#7f6aad"]),
            legend=alt.Legend(title=None, orient="bottom"),
        ),
        tooltip=[
            alt.Tooltip("run:N", title="Run"),
            alt.Tooltip("neutral_threshold:Q", title="Neutral threshold", format=".2f"),
            alt.Tooltip("accuracy:Q", title="Accuracy", format=".2%"),
            alt.Tooltip("balanced_accuracy:Q", title="Balanced accuracy", format=".2%"),
            alt.Tooltip("macro_f1:Q", title="Macro F1", format=".3f"),
        ],
    )
    .properties(height=300)
)

confusion = pd.DataFrame(
    best_details["confusion_matrix"],
    index=["Negative", "Neutral", "Positive"],
    columns=["Negative", "Neutral", "Positive"],
).reset_index(names="Actual")
confusion_long = confusion.melt(id_vars="Actual", var_name="Predicted", value_name="Articles")

confusion_chart = (
    alt.Chart(confusion_long)
    .mark_rect(cornerRadius=3)
    .encode(
        x=alt.X("Predicted:N", title="Predicted"),
        y=alt.Y("Actual:N", title="Actual"),
        color=alt.Color("Articles:Q", scale=alt.Scale(scheme="blues"), legend=None),
        tooltip=["Actual:N", "Predicted:N", "Articles:Q"],
    )
    .properties(height=300)
)
confusion_text = (
    alt.Chart(confusion_long)
    .mark_text(fontSize=16, fontWeight="bold")
    .encode(x="Predicted:N", y="Actual:N", text="Articles:Q", color=alt.value("#1b2633"))
)

bench_left, bench_right = st.columns([1.05, 0.95])
with bench_left:
    st.altair_chart(bench_chart, use_container_width=True)
with bench_right:
    st.altair_chart(confusion_chart + confusion_text, use_container_width=True)

st.subheader("Fast Evidence Table")
display_cols = ["model", "neutral_threshold", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]
st.dataframe(
    unseen_ranking[display_cols].head(8),
    use_container_width=True,
    hide_index=True,
    column_config={
        "model": "Model",
        "neutral_threshold": st.column_config.NumberColumn("Neutral Threshold", format="%.2f"),
        "accuracy": st.column_config.NumberColumn("Accuracy", format="%.1%"),
        "balanced_accuracy": st.column_config.NumberColumn("Balanced Accuracy", format="%.1%"),
        "macro_f1": st.column_config.NumberColumn("Macro F1", format="%.3f"),
        "weighted_f1": st.column_config.NumberColumn("Weighted F1", format="%.3f"),
    },
)

st.subheader("Most Extreme Original Articles")
extreme = pd.concat(
    [
        article_scores.nsmallest(4, "ensemble_tone_score").assign(direction="Lower tone"),
        article_scores.nlargest(4, "ensemble_tone_score").assign(direction="Higher tone"),
    ],
    ignore_index=True,
)
st.dataframe(
    extreme[["direction", "source", "title", "ensemble_tone_score", "model_disagreement"]],
    use_container_width=True,
    hide_index=True,
    column_config={
        "ensemble_tone_score": st.column_config.NumberColumn("Tone", format="%.3f"),
        "model_disagreement": st.column_config.NumberColumn("Disagreement", format="%.3f"),
    },
)
