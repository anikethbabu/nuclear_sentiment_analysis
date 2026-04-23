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
PUBLIC_DIR = ROOT / "models" / "public_now"


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


@st.cache_data(show_spinner=False)
def load_public_now() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    platform = pd.read_csv(PUBLIC_DIR / "public_opinion_by_platform.csv")
    query = pd.read_csv(PUBLIC_DIR / "public_opinion_by_query.csv")
    items = pd.read_csv(PUBLIC_DIR / "public_opinion_scored_items.csv")
    return platform, query, items


counts = load_db_counts()
source_counts = load_sources()
article_scores, source_summary, unseen_ranking, unseen_details = load_outputs()
public_platform, public_query, public_items = load_public_now()

best = unseen_ranking.iloc[0]
best_key = f"{best['model']}|threshold={best['neutral_threshold']}"
best_details = unseen_details[best_key]
overall = source_summary[source_summary["source"] == "ALL"].iloc[0]
by_source = source_summary[source_summary["source"] != "ALL"].copy()
public_overall = public_platform[public_platform["platform"] == "ALL"].iloc[0]
public_by_platform = public_platform[public_platform["platform"] != "ALL"].copy()

st.title("Nuclear Energy Public Sentiment")

st.subheader("Public Opinion Now")
st.markdown(
    f"""
    <div class='small-note'>
    Live directional sample from Google News, Reddit, Hacker News, and GDELT. This is not a statistically representative poll,
    but it shows what current public/media discussion looks like in accessible sources.
    </div>
    """,
    unsafe_allow_html=True,
)

public_cols = st.columns([1, 1, 1, 1])
public_cols[0].metric("Public Items Reviewed", f"{int(public_overall['items']):,}")
public_cols[1].metric("Supportive", f"{public_overall['supportive_share']:.0%}")
public_cols[2].metric("Concerned", f"{public_overall['concerned_share']:.0%}")
public_cols[3].metric("Mixed/Neutral", f"{public_overall['neutral_share']:.0%}")

stance_parts = ["supportive_share", "concerned_share", "neutral_share"]
stance_long = public_by_platform.melt(
    id_vars=["platform", "items"],
    value_vars=stance_parts,
    var_name="stance",
    value_name="share",
)
stance_long["stance"] = stance_long["stance"].map(
    {
        "supportive_share": "Supportive",
        "concerned_share": "Concerned",
        "neutral_share": "Mixed/neutral",
    }
)

stance_chart = (
    alt.Chart(stance_long)
    .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
    .encode(
        x=alt.X("platform:N", title="Public source"),
        y=alt.Y("share:Q", title="Share of items", axis=alt.Axis(format="%")),
        color=alt.Color(
            "stance:N",
            scale=alt.Scale(domain=["Supportive", "Concerned", "Mixed/neutral"], range=["#2374ab", "#b64242", "#8b98a8"]),
            legend=alt.Legend(title=None, orient="bottom"),
        ),
        tooltip=[
            alt.Tooltip("platform:N", title="Source"),
            alt.Tooltip("stance:N", title="Predicted stance"),
            alt.Tooltip("share:Q", title="Share", format=".1%"),
            alt.Tooltip("items:Q", title="Items"),
        ],
    )
    .properties(height=285)
)

query_chart = (
    alt.Chart(public_query)
    .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
    .encode(
        y=alt.Y("query:N", sort="-x", title="Topic"),
        x=alt.X("mean_tone_score:Q", title="Mean public tone", scale=alt.Scale(domain=[-0.65, 0.1])),
        color=alt.Color(
            "mean_tone_score:Q",
            scale=alt.Scale(domain=[-0.6, -0.25, 0], range=["#b64242", "#8b98a8", "#2374ab"]),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("query:N", title="Topic"),
            alt.Tooltip("items:Q", title="Items"),
            alt.Tooltip("mean_tone_score:Q", title="Mean tone", format=".3f"),
            alt.Tooltip("supportive_share:Q", title="Supportive", format=".1%"),
            alt.Tooltip("concerned_share:Q", title="Concerned", format=".1%"),
        ],
    )
    .properties(height=285)
)

public_left, public_right = st.columns([1, 1])
with public_left:
    st.altair_chart(stance_chart, use_container_width=True)
with public_right:
    st.altair_chart(query_chart, use_container_width=True)

left, right = st.columns([1.15, 0.85])

tone_chart = (
    alt.Chart(by_source)
    .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
    .encode(
        y=alt.Y("source:N", sort="-x", title="Source"),
        x=alt.X("mean_tone_score:Q", title="Overall tone", scale=alt.Scale(domain=[-0.15, 0.25])),
        color=alt.Color(
            "mean_tone_score:Q",
            scale=alt.Scale(domain=[-0.12, 0, 0.22], range=["#b64242", "#8b98a8", "#2374ab"]),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("source:N", title="Source"),
            alt.Tooltip("articles:Q", title="Articles"),
            alt.Tooltip("mean_tone_score:Q", title="Tone", format=".3f"),
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

st.subheader("Reliability Check")
st.markdown(
    f"""
    <div class='small-note'>
    This checks the reading system against a small set of human-labeled articles, so the dashboard is not just guessing.
    Best result: <b>{best['accuracy']:.0%}</b> correctly classified.
    </div>
    """,
    unsafe_allow_html=True,
)
rank_view = unseen_ranking.copy()
rank_view["reader"] = rank_view["model"].str.replace(
    "distilbert-base-uncased-finetuned-sst-2-english", "General article reader", regex=False
)
rank_view["reader"] = rank_view["reader"].str.replace(
    "cardiffnlp/twitter-roberta-base-sentiment-latest", "Social media reader", regex=False
)
rank_view["reader"] = rank_view["reader"].str.replace("ProsusAI/finbert", "Finance-news reader", regex=False)
rank_view["reader"] = rank_view["reader"].str.replace("ensemble_mean", "Combined readers", regex=False)
best_by_reader = (
    rank_view.sort_values(["accuracy", "macro_f1"], ascending=False)
    .drop_duplicates("reader")
    .sort_values("accuracy", ascending=False)
)

bench_chart = (
    alt.Chart(best_by_reader)
    .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
    .encode(
        x=alt.X("reader:N", title=None, sort="-y", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("accuracy:Q", title="Correct classifications", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 0.9])),
        color=alt.Color(
            "reader:N",
            scale=alt.Scale(range=["#2374ab", "#e15759", "#59a14f", "#7f6aad"]),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("reader:N", title="Reader"),
            alt.Tooltip("accuracy:Q", title="Correct", format=".2%"),
            alt.Tooltip("balanced_accuracy:Q", title="Fair across groups", format=".2%"),
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

st.subheader("Reader Comparison")
evidence_table = best_by_reader[["reader", "accuracy", "balanced_accuracy", "macro_f1"]].copy()
evidence_table["Correct"] = evidence_table["accuracy"].map(lambda value: f"{value:.1%}")
evidence_table["Fair Across Groups"] = evidence_table["balanced_accuracy"].map(lambda value: f"{value:.1%}")
evidence_table["Overall Balance"] = evidence_table["macro_f1"].map(lambda value: f"{value:.3f}")
evidence_table = evidence_table.rename(columns={"reader": "Reader Type"})[
    ["Reader Type", "Correct", "Fair Across Groups", "Overall Balance"]
]
st.dataframe(
    evidence_table,
    use_container_width=True,
    hide_index=True,
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
    extreme[["direction", "source", "title", "ensemble_tone_score"]],
    use_container_width=True,
    hide_index=True,
    column_config={
        "ensemble_tone_score": st.column_config.NumberColumn("Tone", format="%.3f"),
    },
)
