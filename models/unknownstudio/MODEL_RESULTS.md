# Model Competition Results

## Dataset

- Total imported text articles: 507
- Labeled articles matched from `labels.csv`: 50
- Label counts: `{'Negative': 20, 'Positive': 20, 'Neutral': 10}`

## Cross-validation on labels.csv

| model                      |   accuracy_mean |   accuracy_std |   balanced_accuracy_mean |   balanced_accuracy_std |   f1_macro_mean |   f1_macro_std |   f1_weighted_mean |   f1_weighted_std |   fit_time_mean_sec |
|:---------------------------|----------------:|---------------:|-------------------------:|------------------------:|----------------:|---------------:|-------------------:|------------------:|--------------------:|
| hybrid_tfidf_sgd           |          0.5000 |         0.1000 |                   0.5000 |                  0.1179 |          0.4709 |         0.1300 |             0.4530 |            0.1117 |              0.3950 |
| word_tfidf_logistic        |          0.4800 |         0.1643 |                   0.4500 |                  0.2007 |          0.4199 |         0.2142 |             0.4452 |            0.1796 |              0.0974 |
| hybrid_tfidf_linear_svm    |          0.5200 |         0.1304 |                   0.4500 |                  0.1264 |          0.4085 |         0.1255 |             0.4636 |            0.1126 |              1.3496 |
| soft_voting_ensemble       |          0.4400 |         0.0894 |                   0.4333 |                  0.1236 |          0.3962 |         0.1315 |             0.3820 |            0.0821 |              0.6475 |
| hybrid_tfidf_complement_nb |          0.4000 |         0.1000 |                   0.3500 |                  0.1087 |          0.3063 |         0.1435 |             0.3408 |            0.1214 |              0.1815 |
| word_tfidf_extra_trees     |          0.3600 |         0.1517 |                   0.3000 |                  0.1264 |          0.2552 |         0.1188 |             0.3063 |            0.1425 |              0.4126 |

## Best local model

- Selected model: `hybrid_tfidf_sgd`
- Holdout accuracy: 0.3846
- Holdout balanced accuracy: 0.4222
- Holdout macro F1: 0.3831

## Existing sentiment_cache.csv threshold baseline

- Matched labeled rows: 50
- Accuracy: 0.9800
- Balanced accuracy: 0.9833
- Macro F1: 0.9756

## Imported nuclear.db model baseline

- Mapping used: `pro -> Positive`, `mixed -> Neutral`, `negative -> Negative`
- Accuracy: 0.3600
- Balanced accuracy: 0.3000
- Macro F1: 0.1875

## Caveat

There are only 50 hand-labeled examples, so CV variance matters. The saved model is useful for this article collection, but more labels will make the evaluation much more trustworthy.