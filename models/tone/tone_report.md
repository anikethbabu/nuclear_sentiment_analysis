# Nuclear Article Tone Report

## Method

The SQLite database contains no source-rating or sentiment target column.
No supervised sentiment model is trained from source metadata. Scores are model-estimated continuous tone values in the range -1 to 1.

## Data

- Articles scored: 372
- Models used: `['distilbert-base-uncased-finetuned-sst-2-english', 'ProsusAI/finbert', 'cardiffnlp/twitter-roberta-base-sentiment-latest']`
- Source counts: `{'guardian': 100, 'wnn': 100, 'wna': 100, 'greenpeace': 45, 'beyond_nuclear': 27}`

## Summary Statistics

|   articles |   mean_tone_score |   median_tone_score |   tone_score_std |   q10_tone_score |   q90_tone_score |   mean_model_disagreement | source         |
|-----------:|------------------:|--------------------:|-----------------:|-----------------:|-----------------:|--------------------------:|:---------------|
|        372 |            0.0278 |              0.0332 |           0.1992 |          -0.1663 |           0.3007 |                    0.4161 | ALL            |
|        100 |            0.1976 |              0.2196 |           0.1848 |          -0.1460 |           0.4146 |                    0.3827 | wnn            |
|         45 |            0.0542 |              0.1058 |           0.1657 |          -0.1603 |           0.2359 |                    0.4260 | greenpeace     |
|         27 |            0.0281 |              0.0994 |           0.1654 |          -0.1661 |           0.1660 |                    0.4089 | beyond_nuclear |
|        100 |           -0.0438 |             -0.0931 |           0.1246 |          -0.1638 |           0.1549 |                    0.4123 | wna            |
|        100 |           -0.0825 |             -0.1454 |           0.1809 |          -0.2952 |           0.1722 |                    0.4509 | guardian       |

## Notes

Use the continuous `ensemble_tone_score` and `model_disagreement` columns for analysis. Accuracy is only reported against external ground-truth label files, never against SQLite source metadata.