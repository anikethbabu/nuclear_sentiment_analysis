# Nuclear Sentiment Analysis

This project scores nuclear news/article tone from `nuclear.db`.

Important data rule: the SQLite `articles.label` column is not sentiment ground truth. It is source metadata from the scraper, now handled as `source_type`. The scoring code does not train a classifier from that column and does not report accuracy/CV metrics against it.

## Setup

```powershell
& '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
```

## Score Article Tone

```powershell
& '.\.venv\Scripts\python.exe' score_nuclear_articles.py
```

The scorer uses three pretrained transformer sentiment engines and writes continuous tone statistics to `models/tone/`:

- `article_tone_scores.csv` - per-article continuous scores from each model plus ensemble score
- `model_tone_scores_long.csv` - long-form per-model scoring output
- `tone_summary_by_source.csv` - source-level summary statistics
- `tone_report.md` - readable report

The main statistics are:

- `ensemble_tone_score`: average continuous tone score across models, from `-1` to `1`
- `model_disagreement`: standard deviation across the model scores
- `source_type`: scraper/source metadata only; not used for scoring or statistics

## Import UnknownStudio Articles

The `UnknowStudio4` repository contains additional article text files. If it is checked out next to this repository at `..\UnknowStudio4`, import those articles into this project’s SQLite database with:

```powershell
& '.\.venv\Scripts\python.exe' import_unknownstudio_articles.py
```

That creates or updates an `external_articles` table in `nuclear.db`. It does not create sentiment labels.

## Current Output

The current tone run scored 372 original articles with:

- `distilbert-base-uncased-finetuned-sst-2-english`
- `ProsusAI/finbert`
- `cardiffnlp/twitter-roberta-base-sentiment-latest`

Overall mean `ensemble_tone_score`: `0.0278`

The imported UnknownStudio article table contains 507 rows, with 226 duplicate-content mirrors detected between its two source folders.

## Unseen Accuracy Benchmark

Use the external hand-labeled `UnknowStudio4/labels.csv` articles as an unseen benchmark:

```powershell
& '.\.venv\Scripts\python.exe' evaluate_unseen_sentiment.py
```

Outputs are written to `models/unseen_eval/`.

Best current result on 50 matched unseen labeled articles:

- Model: `distilbert-base-uncased-finetuned-sst-2-english`
- Neutral threshold: `0.05`
- Accuracy: `0.8200`
- Balanced accuracy: `0.8167`
- Macro F1: `0.8001`

These are real benchmark metrics because they compare against the external label file, not the SQLite source-type metadata.

## Presentation Dashboard

Launch the Streamlit dashboard:

```powershell
& '.\.venv\Scripts\python.exe' -m streamlit run dashboard.py
```

The dashboard is designed as a nontechnical 1-2 minute briefing: corrected database counts, source-level tone findings, current public-source sentiment, reliability check, representative extreme-tone articles, and a concise narrative about how sentiment can affect public support, permitting, investment, and policy momentum for nuclear energy.

## Public Opinion Now

Collect current public-source signals and estimate present stance:

```powershell
& '.\.venv\Scripts\python.exe' public_opinion_now.py
```

The live collector pulls accessible public content from Google News RSS, Reddit search, Hacker News, and GDELT when reachable. X/Twitter is not collected without API credentials. Outputs are written to `models/public_now/` and a `public_opinion_items` table is written into `nuclear.db`.

Current live directional sample:

- Items scored: `499`
- Predicted supportive signal: `27.3%`
- Predicted concerned signal: `57.5%`
- Predicted mixed/neutral signal: `15.2%`
- Most concern-heavy topic in this pull: `nuclear waste`

This is a directional public/media signal, not a statistically representative poll.
