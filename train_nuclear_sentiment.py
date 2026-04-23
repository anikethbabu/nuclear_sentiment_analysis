from __future__ import annotations

import argparse
import json
import os
import random
import re
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC


RANDOM_STATE = 42
DEFAULT_DB = "nuclear.db"
LABEL_ORDER = ["negative", "mixed", "pro"]


@dataclass
class RunConfig:
    db_path: str
    output_dir: str
    table: str
    text_limit: int
    cv_folds: int
    run_transformer: bool
    transformer_model: str
    transformer_epochs: int
    transformer_batch_size: int
    created_at_epoch: float


def set_reproducible(seed: int = RANDOM_STATE) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_articles(db_path: str, table: str = "articles", text_limit: int = 15000) -> pd.DataFrame:
    con = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table}", con)
    finally:
        con.close()

    required = {"id", "source", "label", "title", "url", "content"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {table}: {missing}")

    df = df.dropna(subset=["label"]).copy()
    df["title"] = df["title"].map(normalize_text)
    df["content"] = df["content"].map(normalize_text)
    df["text"] = (df["title"] + ". " + df["content"]).map(normalize_text)
    df["text"] = df["text"].str.slice(0, text_limit)
    df = df[df["text"].str.len() > 20].drop_duplicates(subset=["url"]).reset_index(drop=True)
    df["label"] = df["label"].str.strip().str.lower()
    return df


def make_models() -> dict[str, Pipeline | VotingClassifier]:
    word_tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.92,
        max_features=90000,
        sublinear_tf=True,
        strip_accents="unicode",
        stop_words="english",
    )
    char_tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=120000,
        sublinear_tf=True,
    )
    hybrid_features = FeatureUnion(
        [
            ("word", clone(word_tfidf)),
            ("char", clone(char_tfidf)),
        ],
        n_jobs=-1,
    )

    logistic = Pipeline(
        [
            ("tfidf", clone(word_tfidf)),
            (
                "clf",
                LogisticRegression(
                    C=4.0,
                    class_weight="balanced",
                    solver="saga",
                    max_iter=5000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    svm = Pipeline(
        [
            ("features", clone(hybrid_features)),
            (
                "clf",
                CalibratedClassifierCV(
                    estimator=LinearSVC(
                        C=1.5,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        dual="auto",
                        max_iter=10000,
                    ),
                    method="sigmoid",
                    cv=3,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    nb = Pipeline(
        [
            ("features", clone(hybrid_features)),
            ("clf", ComplementNB(alpha=0.2)),
        ]
    )

    sgd = Pipeline(
        [
            ("features", clone(hybrid_features)),
            (
                "clf",
                SGDClassifier(
                    loss="modified_huber",
                    alpha=1e-5,
                    penalty="elasticnet",
                    l1_ratio=0.08,
                    class_weight="balanced",
                    max_iter=3000,
                    tol=1e-4,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    forest = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=40000,
                    sublinear_tf=True,
                    stop_words="english",
                ),
            ),
            (
                "clf",
                ExtraTreesClassifier(
                    n_estimators=700,
                    class_weight="balanced",
                    max_features="sqrt",
                    min_samples_leaf=2,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    ensemble = VotingClassifier(
        estimators=[
            ("logistic", clone(logistic)),
            ("svm", clone(svm)),
            ("nb", clone(nb)),
            ("sgd", clone(sgd)),
        ],
        voting="soft",
        weights=[2, 3, 1, 2],
        n_jobs=-1,
    )

    return {
        "word_tfidf_logistic": logistic,
        "hybrid_tfidf_linear_svm": svm,
        "hybrid_tfidf_complement_nb": nb,
        "hybrid_tfidf_sgd": sgd,
        "word_tfidf_extra_trees": forest,
        "soft_voting_ensemble": ensemble,
    }


def evaluate_cv(models: dict[str, Any], x: pd.Series, y: pd.Series, cv_folds: int) -> pd.DataFrame:
    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "f1_macro": "f1_macro",
        "f1_weighted": "f1_weighted",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
    }
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    rows: list[dict[str, Any]] = []
    for name, model in models.items():
        print(f"\n[CV] {name}")
        scores = cross_validate(
            model,
            x,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            return_train_score=True,
            error_score="raise",
        )
        row: dict[str, Any] = {"model": name}
        for metric in scoring:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1))
        row["fit_time_mean_sec"] = float(np.mean(scores["fit_time"]))
        row["score_time_mean_sec"] = float(np.mean(scores["score_time"]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["f1_macro_mean", "balanced_accuracy_mean"], ascending=False)


def evaluate_holdout(model: Any, x_train: pd.Series, x_test: pd.Series, y_train: pd.Series, y_test: pd.Series) -> dict[str, Any]:
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    report = classification_report(y_test, pred, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "f1_macro": float(f1_score(y_test, pred, average="macro")),
        "report": report,
        "confusion_matrix": confusion_matrix(y_test, pred, labels=LABEL_ORDER).tolist(),
        "predictions": pred,
    }


def save_confusion_matrix(cm: list[list[int]], labels: list[str], path: Path, title: str) -> None:
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def get_top_features(model: Any, labels: list[str], top_n: int = 25) -> pd.DataFrame:
    if not isinstance(model, Pipeline):
        return pd.DataFrame()
    if "tfidf" not in model.named_steps or "clf" not in model.named_steps:
        return pd.DataFrame()
    clf = model.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return pd.DataFrame()
    features = np.asarray(model.named_steps["tfidf"].get_feature_names_out())
    rows: list[dict[str, Any]] = []
    for class_index, label in enumerate(labels):
        weights = clf.coef_[class_index]
        best = np.argsort(weights)[-top_n:][::-1]
        for rank, idx in enumerate(best, start=1):
            rows.append({"label": label, "rank": rank, "feature": features[idx], "weight": float(weights[idx])})
    return pd.DataFrame(rows)


def predict_full_dataset(model: Any, df: pd.DataFrame) -> pd.DataFrame:
    pred = model.predict(df["text"])
    out = df[["id", "source", "label", "title", "url"]].copy()
    out["predicted_label"] = pred
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(df["text"])
        classes = list(model.classes_)
        for i, klass in enumerate(classes):
            out[f"prob_{klass}"] = probs[:, i]
        out["confidence"] = probs.max(axis=1)
    return out


def run_transformer_holdout(
    df: pd.DataFrame,
    output_dir: Path,
    model_name: str,
    epochs: int,
    batch_size: int,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    class TextDataset(Dataset):
        def __init__(self, texts: list[str], labels: np.ndarray | None) -> None:
            self.texts = texts
            self.labels = labels

        def __len__(self) -> int:
            return len(self.texts)

        def __getitem__(self, idx: int) -> dict[str, Any]:
            item = {"text": self.texts[idx]}
            if self.labels is not None:
                item["label"] = int(self.labels[idx])
            return item

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        enc = tokenizer(
            [b["text"] for b in batch],
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt",
        )
        if "label" in batch[0]:
            enc["labels"] = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        return enc

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["label"])
    x_train, x_test, y_train, y_test = train_test_split(
        df["text"].tolist(),
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    id2label = {i: label for i, label in enumerate(label_encoder.classes_)}
    label2id = {label: i for i, label in id2label.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_encoder.classes_),
        id2label=id2label,
        label2id=label2id,
    ).to(device)

    train_loader = DataLoader(TextDataset(x_train, y_train), batch_size=batch_size, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(TextDataset(x_test, y_test), batch_size=batch_size, shuffle=False, collate_fn=collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    steps = max(1, len(train_loader) * epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max(1, steps // 10), num_training_steps=steps)

    model.train()
    for epoch in range(epochs):
        losses = []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            out = model(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(float(out.loss.detach().cpu()))
        print(f"[Transformer] epoch {epoch + 1}/{epochs} loss={np.mean(losses):.4f}")

    model.eval()
    preds: list[int] = []
    refs: list[int] = []
    with torch.no_grad():
        for batch in test_loader:
            labels = batch.pop("labels").numpy().tolist()
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds.extend(torch.argmax(logits, dim=-1).cpu().numpy().tolist())
            refs.extend(labels)

    pred_labels = label_encoder.inverse_transform(preds)
    ref_labels = label_encoder.inverse_transform(refs)
    cm = confusion_matrix(ref_labels, pred_labels, labels=list(label_encoder.classes_)).tolist()
    transformer_dir = output_dir / "transformer_distilbert"
    transformer_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(transformer_dir)
    tokenizer.save_pretrained(transformer_dir)

    return {
        "model": model_name,
        "device": str(device),
        "epochs": epochs,
        "batch_size": batch_size,
        "accuracy": float(accuracy_score(ref_labels, pred_labels)),
        "balanced_accuracy": float(balanced_accuracy_score(ref_labels, pred_labels)),
        "f1_macro": float(f1_score(ref_labels, pred_labels, average="macro")),
        "classification_report": classification_report(ref_labels, pred_labels, output_dict=True, zero_division=0),
        "confusion_matrix": cm,
        "labels": list(label_encoder.classes_),
        "saved_model_dir": str(transformer_dir),
    }


def write_markdown_report(
    output_dir: Path,
    df: pd.DataFrame,
    cv_results: pd.DataFrame,
    holdout_results: dict[str, Any],
    best_model_name: str,
    transformer_results: dict[str, Any] | None,
) -> None:
    label_counts = df["label"].value_counts().to_dict()
    source_counts = df["source"].value_counts().to_dict()
    lines = [
        "# Nuclear Sentiment Model Report",
        "",
        "## Data caveat",
        "",
        "The database labels are weak labels assigned by source, not independent human annotations.",
        "This is useful for modeling article stance in this collection, but it can overestimate true sentiment accuracy because source and label are perfectly linked.",
        "",
        "## Dataset",
        "",
        f"- Articles after cleaning: {len(df)}",
        f"- Label counts: `{label_counts}`",
        f"- Source counts: `{source_counts}`",
        f"- Median text length: {int(df['text'].str.len().median())} characters",
        "",
        "## Cross-validation ranking",
        "",
        cv_results.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Best classical model",
        "",
        f"Selected by macro F1: `{best_model_name}`",
        "",
        "Holdout metrics:",
        "",
        f"- Accuracy: {holdout_results['accuracy']:.4f}",
        f"- Balanced accuracy: {holdout_results['balanced_accuracy']:.4f}",
        f"- Macro F1: {holdout_results['f1_macro']:.4f}",
        "",
        "Classification report:",
        "",
        "```json",
        json.dumps(holdout_results["report"], indent=2),
        "```",
    ]
    if transformer_results is not None:
        lines.extend(
            [
                "",
                "## Transformer holdout",
                "",
                f"- Base model: `{transformer_results['model']}`",
                f"- Device: `{transformer_results['device']}`",
                f"- Accuracy: {transformer_results['accuracy']:.4f}",
                f"- Balanced accuracy: {transformer_results['balanced_accuracy']:.4f}",
                f"- Macro F1: {transformer_results['f1_macro']:.4f}",
                f"- Saved model: `{transformer_results['saved_model_dir']}`",
            ]
        )
    (output_dir / "model_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train nuclear article sentiment/stance models from SQLite.")
    parser.add_argument("--db", default=DEFAULT_DB, help="Path to SQLite database.")
    parser.add_argument("--table", default="articles", help="SQLite table name.")
    parser.add_argument("--output", default="models", help="Output directory for reports and trained models.")
    parser.add_argument("--text-limit", type=int, default=15000, help="Max characters per article for classical models.")
    parser.add_argument("--cv-folds", type=int, default=5, help="Stratified CV folds.")
    parser.add_argument("--run-transformer", action="store_true", help="Also fine-tune a DistilBERT classifier on holdout split.")
    parser.add_argument("--transformer-model", default="distilbert/distilbert-base-uncased", help="Hugging Face model id.")
    parser.add_argument("--transformer-epochs", type=int, default=3)
    parser.add_argument("--transformer-batch-size", type=int, default=8)
    args = parser.parse_args()

    set_reproducible()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)

    config = RunConfig(
        db_path=args.db,
        output_dir=args.output,
        table=args.table,
        text_limit=args.text_limit,
        cv_folds=args.cv_folds,
        run_transformer=args.run_transformer,
        transformer_model=args.transformer_model,
        transformer_epochs=args.transformer_epochs,
        transformer_batch_size=args.transformer_batch_size,
        created_at_epoch=time.time(),
    )
    (output_dir / "run_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    df = load_articles(args.db, args.table, args.text_limit)
    df.to_csv(output_dir / "clean_articles.csv", index=False)
    print(f"Loaded {len(df)} articles")
    print(df["label"].value_counts())

    cv_folds = min(args.cv_folds, int(df["label"].value_counts().min()))
    models = make_models()
    cv_results = evaluate_cv(models, df["text"], df["label"], cv_folds=cv_folds)
    cv_results.to_csv(output_dir / "cv_results.csv", index=False)
    print("\nCV results")
    print(cv_results)

    best_model_name = str(cv_results.iloc[0]["model"])
    best_template = models[best_model_name]
    x_train, x_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )
    holdout = evaluate_holdout(clone(best_template), x_train, x_test, y_train, y_test)
    save_confusion_matrix(
        holdout["confusion_matrix"],
        LABEL_ORDER,
        output_dir / "plots" / "best_model_confusion_matrix.png",
        f"{best_model_name} holdout confusion matrix",
    )
    holdout_public = {k: v for k, v in holdout.items() if k != "predictions"}
    (output_dir / "holdout_report.json").write_text(json.dumps(holdout_public, indent=2), encoding="utf-8")

    final_model = clone(best_template)
    final_model.fit(df["text"], df["label"])
    joblib.dump(final_model, output_dir / "best_nuclear_sentiment_model.joblib")
    predictions = predict_full_dataset(final_model, df)
    predictions.to_csv(output_dir / "article_predictions.csv", index=False)

    top_features = get_top_features(final_model, sorted(df["label"].unique()))
    if not top_features.empty:
        top_features.to_csv(output_dir / "top_features.csv", index=False)

    transformer_results = None
    if args.run_transformer:
        transformer_results = run_transformer_holdout(
            df,
            output_dir,
            args.transformer_model,
            args.transformer_epochs,
            args.transformer_batch_size,
        )
        (output_dir / "transformer_holdout_report.json").write_text(
            json.dumps(transformer_results, indent=2),
            encoding="utf-8",
        )
        save_confusion_matrix(
            transformer_results["confusion_matrix"],
            transformer_results["labels"],
            output_dir / "plots" / "transformer_confusion_matrix.png",
            "DistilBERT holdout confusion matrix",
        )

    write_markdown_report(output_dir, df, cv_results, holdout_public, best_model_name, transformer_results)
    print(f"\nDone. Report: {output_dir / 'model_report.md'}")
    print(f"Best model: {best_model_name}")


if __name__ == "__main__":
    main()
