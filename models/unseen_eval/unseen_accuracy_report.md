# Unseen Accuracy Report

## Benchmark

This evaluates pretrained sentiment scorers on the unseen labeled articles from `UnknowStudio4/labels.csv`.
No model is trained on the benchmark labels in this script.

- Matched labeled articles: 50
- Label counts: `{'Negative': 20, 'Positive': 20, 'Neutral': 10}`

## Ranking

| model                                            |   neutral_threshold |   accuracy |   balanced_accuracy |   macro_f1 |   weighted_f1 |   matched_rows |
|:-------------------------------------------------|--------------------:|-----------:|--------------------:|-----------:|--------------:|---------------:|
| distilbert-base-uncased-finetuned-sst-2-english  |              0.0500 |     0.8200 |              0.8167 |     0.8001 |        0.8210 |             50 |
| distilbert-base-uncased-finetuned-sst-2-english  |              0.1000 |     0.8200 |              0.8167 |     0.7978 |        0.8241 |             50 |
| distilbert-base-uncased-finetuned-sst-2-english  |              0.2000 |     0.8000 |              0.8000 |     0.7800 |        0.8080 |             50 |
| distilbert-base-uncased-finetuned-sst-2-english  |              0.3000 |     0.7800 |              0.7833 |     0.7624 |        0.7918 |             50 |
| ensemble_mean                                    |              0.2000 |     0.6800 |              0.6833 |     0.6581 |        0.6898 |             50 |
| ensemble_mean                                    |              0.1000 |     0.7000 |              0.6333 |     0.6302 |        0.6931 |             50 |
| ensemble_mean                                    |              0.3000 |     0.6200 |              0.6333 |     0.6090 |        0.6433 |             50 |
| ensemble_mean                                    |              0.0500 |     0.7000 |              0.6000 |     0.5771 |        0.6659 |             50 |
| cardiffnlp/twitter-roberta-base-sentiment-latest |              0.3000 |     0.3800 |              0.4667 |     0.3325 |        0.3207 |             50 |
| cardiffnlp/twitter-roberta-base-sentiment-latest |              0.0500 |     0.3600 |              0.4000 |     0.3272 |        0.3326 |             50 |
| cardiffnlp/twitter-roberta-base-sentiment-latest |              0.1000 |     0.3600 |              0.4000 |     0.3272 |        0.3326 |             50 |
| cardiffnlp/twitter-roberta-base-sentiment-latest |              0.2000 |     0.3400 |              0.3833 |     0.2989 |        0.3016 |             50 |
| ProsusAI/finbert                                 |              0.3000 |     0.2800 |              0.3000 |     0.2862 |        0.3054 |             50 |
| ProsusAI/finbert                                 |              0.2000 |     0.2800 |              0.2833 |     0.2806 |        0.3059 |             50 |
| ProsusAI/finbert                                 |              0.0500 |     0.2800 |              0.2833 |     0.2777 |        0.3017 |             50 |
| ProsusAI/finbert                                 |              0.1000 |     0.2800 |              0.2833 |     0.2777 |        0.3017 |             50 |

## Best Run

- Model: `distilbert-base-uncased-finetuned-sst-2-english`
- Neutral threshold: `0.05`
- Accuracy: `0.8200`
- Balanced accuracy: `0.8167`
- Macro F1: `0.8001`

Confusion matrix label order: `Negative`, `Neutral`, `Positive`

```json
[
  [
    14,
    4,
    2
  ],
  [
    1,
    8,
    1
  ],
  [
    0,
    1,
    19
  ]
]
```

Classification report:

```json
{
  "Negative": {
    "precision": 0.9333333333333333,
    "recall": 0.7,
    "f1-score": 0.8,
    "support": 20.0
  },
  "Neutral": {
    "precision": 0.6153846153846154,
    "recall": 0.8,
    "f1-score": 0.6956521739130435,
    "support": 10.0
  },
  "Positive": {
    "precision": 0.8636363636363636,
    "recall": 0.95,
    "f1-score": 0.9047619047619048,
    "support": 20.0
  },
  "accuracy": 0.82,
  "macro avg": {
    "precision": 0.8041181041181041,
    "recall": 0.8166666666666668,
    "f1-score": 0.8001380262249828,
    "support": 50.0
  },
  "weighted avg": {
    "precision": 0.841864801864802,
    "recall": 0.82,
    "f1-score": 0.8210351966873705,
    "support": 50.0
  }
}
```