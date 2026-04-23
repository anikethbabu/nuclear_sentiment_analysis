# Nuclear Sentiment Analysis

This project trains nuclear article sentiment/stance models from `nuclear.db`.

Important caveat: the current database uses weak labels from source identity:

- `pro`: World Nuclear News and World Nuclear Association
- `negative`: Greenpeace and Beyond Nuclear
- `mixed`: The Guardian

That makes the trained model useful for modeling stance in this scraped corpus, but it is not a substitute for human-labeled sentiment data.

## Setup

```powershell
& '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
```

## Train classical CV models

```powershell
& '.\.venv\Scripts\python.exe' train_nuclear_sentiment.py
```

The script trains and cross-validates:

- TF-IDF + logistic regression
- Hybrid word/character TF-IDF + calibrated linear SVM
- Hybrid word/character TF-IDF + Complement Naive Bayes
- Hybrid word/character TF-IDF + SGD classifier
- Word TF-IDF + Extra Trees
- A soft-voting ensemble

Outputs are written to `models/`.

## Train with transformer fine-tuning too

```powershell
& '.\.venv\Scripts\python.exe' train_nuclear_sentiment.py --run-transformer --transformer-epochs 3
```

This additionally fine-tunes `distilbert/distilbert-base-uncased` using PyTorch and Hugging Face Transformers.

## Predict a new article

```powershell
& '.\.venv\Scripts\python.exe' predict_nuclear_sentiment.py --text "New nuclear reactor approvals are expected to reduce emissions and stabilize the grid."
```

## Import and compare UnknownStudio articles

The `UnknowStudio4` repository contains 507 article text entries and 50 hand labels. If that repo is checked out next to this one at `..\UnknowStudio4`, import its articles into SQLite with:

```powershell
& '.\.venv\Scripts\python.exe' import_unknownstudio_articles.py
```

Run the model competition against its labeled set:

```powershell
& '.\.venv\Scripts\python.exe' train_unknownstudio_competition.py
```

Outputs are written to `models/unknownstudio/`, including the imported SQLite database, predictions for every imported article, CV metrics, and `unknownstudio_best_sentiment_model.joblib`.

Current comparison summary:

- Existing `sentiment_cache.csv` threshold baseline: macro F1 `0.9756`
- Best new model trained only on the 50 labels: CV macro F1 `0.4709`
- Transferred `nuclear.db` model mapped to Positive/Neutral/Negative: macro F1 `0.1875`

The cached baseline is currently strongest on those 50 labels. The transferred `nuclear.db` model performs poorly because its source-derived `pro/mixed/negative` labels do not match the hand-labeled Positive/Neutral/Negative task very well.
