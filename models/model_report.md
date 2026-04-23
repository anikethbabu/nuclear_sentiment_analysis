# Nuclear Sentiment Model Report

## Data caveat

The database labels are weak labels assigned by source, not independent human annotations.
This is useful for modeling article stance in this collection, but it can overestimate true sentiment accuracy because source and label are perfectly linked.

## Dataset

- Articles after cleaning: 372
- Label counts: `{'pro': 200, 'mixed': 100, 'negative': 72}`
- Source counts: `{'guardian': 100, 'wnn': 100, 'wna': 100, 'greenpeace': 45, 'beyond_nuclear': 27}`
- Median text length: 5014 characters

## Cross-validation ranking

| model                      |   accuracy_mean |   accuracy_std |   balanced_accuracy_mean |   balanced_accuracy_std |   f1_macro_mean |   f1_macro_std |   f1_weighted_mean |   f1_weighted_std |   precision_macro_mean |   precision_macro_std |   recall_macro_mean |   recall_macro_std |   fit_time_mean_sec |   score_time_mean_sec |
|:---------------------------|----------------:|---------------:|-------------------------:|------------------------:|----------------:|---------------:|-------------------:|------------------:|-----------------------:|----------------------:|--------------------:|-------------------:|--------------------:|----------------------:|
| hybrid_tfidf_sgd           |          0.9946 |         0.0074 |                   0.9950 |                  0.0075 |          0.9950 |         0.0069 |             0.9946 |            0.0074 |                 0.9952 |                0.0071 |              0.9950 |             0.0075 |              2.7033 |                0.5502 |
| hybrid_tfidf_linear_svm    |          0.9920 |         0.0119 |                   0.9900 |                  0.0149 |          0.9923 |         0.0115 |             0.9919 |            0.0121 |                 0.9952 |                0.0071 |              0.9900 |             0.0149 |              3.0842 |                0.7749 |
| soft_voting_ensemble       |          0.9839 |         0.0112 |                   0.9817 |                  0.0124 |          0.9848 |         0.0106 |             0.9838 |            0.0113 |                 0.9886 |                0.0094 |              0.9817 |             0.0124 |              4.0889 |                1.7595 |
| word_tfidf_logistic        |          0.9785 |         0.0205 |                   0.9733 |                  0.0253 |          0.9792 |         0.0201 |             0.9780 |            0.0212 |                 0.9875 |                0.0114 |              0.9733 |             0.0253 |              0.9558 |                0.0837 |
| word_tfidf_extra_trees     |          0.9677 |         0.0154 |                   0.9600 |                  0.0190 |          0.9688 |         0.0153 |             0.9671 |            0.0160 |                 0.9813 |                0.0084 |              0.9600 |             0.0190 |              1.0590 |                0.1431 |
| hybrid_tfidf_complement_nb |          0.9597 |         0.0213 |                   0.9475 |                  0.0274 |          0.9581 |         0.0243 |             0.9588 |            0.0224 |                 0.9740 |                0.0170 |              0.9475 |             0.0274 |              1.7380 |                0.5656 |

## Best classical model

Selected by macro F1: `hybrid_tfidf_sgd`

Holdout metrics:

- Accuracy: 1.0000
- Balanced accuracy: 1.0000
- Macro F1: 1.0000

Classification report:

```json
{
  "mixed": {
    "precision": 1.0,
    "recall": 1.0,
    "f1-score": 1.0,
    "support": 20.0
  },
  "negative": {
    "precision": 1.0,
    "recall": 1.0,
    "f1-score": 1.0,
    "support": 15.0
  },
  "pro": {
    "precision": 1.0,
    "recall": 1.0,
    "f1-score": 1.0,
    "support": 40.0
  },
  "accuracy": 1.0,
  "macro avg": {
    "precision": 1.0,
    "recall": 1.0,
    "f1-score": 1.0,
    "support": 75.0
  },
  "weighted avg": {
    "precision": 1.0,
    "recall": 1.0,
    "f1-score": 1.0,
    "support": 75.0
  }
}
```

## Transformer holdout

- Base model: `distilbert/distilbert-base-uncased`
- Device: `cpu`
- Accuracy: 0.9867
- Balanced accuracy: 0.9778
- Macro F1: 0.9804
- Saved model: `models\transformer_distilbert`