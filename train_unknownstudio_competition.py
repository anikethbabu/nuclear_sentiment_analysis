from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


RANDOM_STATE = 42
ARTICLE_FOLDERS = {
    "ANS": Path("ans_articles"),
    "World Nuclear": Path("World_Nuclear_Scraper") / "articles",
}
LABEL_ORDER = ["Negative", "Neutral", "Positive"]


def clean_text(text: str) -> str:
    text = text.replace("\ufeff", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_articles(external_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source, relative_folder in ARTICLE_FOLDERS.items():
        folder = external_root / relative_folder
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.txt")):
            text = clean_text(path.read_text(encoding="utf-8", errors="replace"))
            if not text:
                continue
            rows.append(
                {
                    "article_id": f"{source}::{path.name}",
                    "source": source,
                    "filename": path.name,
                    "title": path.stem.replace("-", " "),
                    "text": text,
                    "word_count": len(text.split()),
                }
            )
    return pd.DataFrame(rows).drop_duplicates("article_id").reset_index(drop=True)


def load_labeled_frame(labels_path: Path, external_root: Path) -> pd.DataFrame:
    articles = load_articles(external_root)
    labels = pd.read_csv(labels_path)
    labels["label"] = labels["label"].str.strip().str.title()
    frame = articles.merge(labels, on="article_id", how="inner")
    if frame.empty:
        raise ValueError("No labeled articles matched labels.csv")
    return frame


def make_models() -> dict[str, Any]:
    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        max_features=60000,
        sublinear_tf=True,
        strip_accents="unicode",
        stop_words="english",
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        max_features=90000,
        sublinear_tf=True,
    )
    hybrid = FeatureUnion([("word", clone(word)), ("char", clone(char))], n_jobs=-1)

    logistic = Pipeline(
        [
            ("tfidf", clone(word)),
            (
                "clf",
                LogisticRegression(
                    C=2.5,
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
            ("features", clone(hybrid)),
            (
                "clf",
                CalibratedClassifierCV(
                    estimator=LinearSVC(
                        C=1.0,
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
    nb = Pipeline([("features", clone(hybrid)), ("clf", ComplementNB(alpha=0.25))])
    sgd = Pipeline(
        [
            ("features", clone(hybrid)),
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
    trees = Pipeline(
        [
            ("tfidf", clone(word)),
            (
                "clf",
                ExtraTreesClassifier(
                    n_estimators=500,
                    class_weight="balanced",
                    max_features="sqrt",
                    min_samples_leaf=1,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    ensemble = VotingClassifier(
        estimators=[("logistic", clone(logistic)), ("svm", clone(svm)), ("nb", clone(nb)), ("sgd", clone(sgd))],
        voting="soft",
        weights=[2, 3, 1, 2],
        n_jobs=-1,
    )
    return {
        "word_tfidf_logistic": logistic,
        "hybrid_tfidf_linear_svm": svm,
        "hybrid_tfidf_complement_nb": nb,
        "hybrid_tfidf_sgd": sgd,
        "word_tfidf_extra_trees": trees,
        "soft_voting_ensemble": ensemble,
    }


def evaluate_cache(labels_path: Path, cache_path: Path) -> dict[str, Any] | None:
    if not cache_path.exists():
        return None
    labels = pd.read_csv(labels_path)
    cache = pd.read_csv(cache_path).rename(columns={"filename": "article_id"})
    merged = labels.merge(cache, on="article_id", how="inner")
    if merged.empty:
        return None

    def score_to_label(score: float) -> str:
        if score > 0.2:
            return "Positive"
        if score < -0.2:
            return "Negative"
        return "Neutral"

    y_true = merged["label"].str.title()
    y_pred = merged["score"].map(score_to_label)
    return metric_block(y_true, y_pred, labels=LABEL_ORDER) | {"matched_rows": int(len(merged))}


def evaluate_external_model(model_path: Path, frame: pd.DataFrame) -> dict[str, Any] | None:
    if not model_path.exists():
        return None
    model = joblib.load(model_path)
    raw_pred = model.predict(frame["text"])
    mapping = {"pro": "Positive", "mixed": "Neutral", "negative": "Negative"}
    y_pred = pd.Series(raw_pred).map(lambda x: mapping.get(str(x).lower(), str(x).title()))
    return metric_block(frame["label"], y_pred, labels=LABEL_ORDER)


def metric_block(y_true: pd.Series, y_pred: pd.Series, labels: list[str]) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "classification_report": classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def cross_validate_models(models: dict[str, Any], frame: pd.DataFrame, folds: int) -> pd.DataFrame:
    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "f1_macro": "f1_macro",
        "f1_weighted": "f1_weighted",
    }
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, model in models.items():
        print(f"[CV] {name}")
        scores = cross_validate(model, frame["text"], frame["label"], cv=cv, scoring=scoring, n_jobs=1, error_score="raise")
        row = {"model": name}
        for metric in scoring:
            row[f"{metric}_mean"] = float(np.mean(scores[f"test_{metric}"]))
            row[f"{metric}_std"] = float(np.std(scores[f"test_{metric}"], ddof=1))
        row["fit_time_mean_sec"] = float(np.mean(scores["fit_time"]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["f1_macro_mean", "balanced_accuracy_mean"], ascending=False)


def write_report(
    output: Path,
    labeled: pd.DataFrame,
    all_articles: pd.DataFrame,
    cv_results: pd.DataFrame,
    holdout: dict[str, Any],
    best_model: str,
    cache_eval: dict[str, Any] | None,
    external_eval: dict[str, Any] | None,
) -> None:
    lines = [
        "# Model Competition Results",
        "",
        "## Dataset",
        "",
        f"- Total imported text articles: {len(all_articles)}",
        f"- Labeled articles matched from `labels.csv`: {len(labeled)}",
        f"- Label counts: `{labeled['label'].value_counts().to_dict()}`",
        "",
        "## Cross-validation on labels.csv",
        "",
        cv_results.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Best local model",
        "",
        f"- Selected model: `{best_model}`",
        f"- Holdout accuracy: {holdout['accuracy']:.4f}",
        f"- Holdout balanced accuracy: {holdout['balanced_accuracy']:.4f}",
        f"- Holdout macro F1: {holdout['f1_macro']:.4f}",
    ]
    if cache_eval:
        lines.extend(
            [
                "",
                "## Existing sentiment_cache.csv threshold baseline",
                "",
                f"- Matched labeled rows: {cache_eval['matched_rows']}",
                f"- Accuracy: {cache_eval['accuracy']:.4f}",
                f"- Balanced accuracy: {cache_eval['balanced_accuracy']:.4f}",
                f"- Macro F1: {cache_eval['f1_macro']:.4f}",
            ]
        )
    if external_eval:
        lines.extend(
            [
                "",
                "## Imported nuclear.db model baseline",
                "",
                "- Mapping used: `pro -> Positive`, `mixed -> Neutral`, `negative -> Negative`",
                f"- Accuracy: {external_eval['accuracy']:.4f}",
                f"- Balanced accuracy: {external_eval['balanced_accuracy']:.4f}",
                f"- Macro F1: {external_eval['f1_macro']:.4f}",
            ]
        )
    lines.extend(
        [
            "",
            "## Caveat",
            "",
            "There are only 50 hand-labeled examples, so CV variance matters. The saved model is useful for this article collection, but more labels will make the evaluation much more trustworthy.",
        ]
    )
    (output / "MODEL_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and compare nuclear sentiment models for this repo.")
    parser.add_argument("--external-root", default=str(Path("..") / "UnknowStudio4"))
    parser.add_argument("--labels", default="")
    parser.add_argument("--cache", default="")
    parser.add_argument("--external-model", default="models/best_nuclear_sentiment_model.joblib")
    parser.add_argument("--output", default="models/unknownstudio")
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    external_root = Path(args.external_root)
    labels_path = Path(args.labels) if args.labels else external_root / "labels.csv"
    cache_path = Path(args.cache) if args.cache else external_root / "sentiment_cache.csv"

    all_articles = load_articles(external_root)
    labeled = load_labeled_frame(labels_path, external_root)
    folds = min(args.cv_folds, int(labeled["label"].value_counts().min()))
    models = make_models()

    cv_results = cross_validate_models(models, labeled, folds)
    cv_results.to_csv(output / "unknownstudio_cv_results.csv", index=False)

    best_name = str(cv_results.iloc[0]["model"])
    x_train, x_test, y_train, y_test = train_test_split(
        labeled["text"], labeled["label"], test_size=0.25, random_state=RANDOM_STATE, stratify=labeled["label"]
    )
    best_holdout_model = clone(models[best_name])
    best_holdout_model.fit(x_train, y_train)
    holdout_pred = best_holdout_model.predict(x_test)
    holdout = metric_block(y_test, pd.Series(holdout_pred), labels=LABEL_ORDER)
    (output / "unknownstudio_holdout_report.json").write_text(json.dumps(holdout, indent=2), encoding="utf-8")

    final_model = clone(models[best_name])
    final_model.fit(labeled["text"], labeled["label"])
    joblib.dump(final_model, output / "unknownstudio_best_sentiment_model.joblib")

    all_predictions = all_articles[["article_id", "source", "filename", "title", "word_count"]].copy()
    all_predictions["predicted_label"] = final_model.predict(all_articles["text"])
    if hasattr(final_model, "predict_proba"):
        probs = final_model.predict_proba(all_articles["text"])
        for index, label in enumerate(final_model.classes_):
            all_predictions[f"prob_{label}"] = probs[:, index]
        all_predictions["confidence"] = probs.max(axis=1)
    all_predictions.to_csv(output / "unknownstudio_article_predictions.csv", index=False)

    cache_eval = evaluate_cache(labels_path, cache_path)
    if cache_eval:
        (output / "sentiment_cache_eval.json").write_text(json.dumps(cache_eval, indent=2), encoding="utf-8")

    external_eval = None
    if args.external_model:
        external_eval = evaluate_external_model(Path(args.external_model), labeled)
        if external_eval:
            (output / "imported_nuclear_db_model_eval.json").write_text(json.dumps(external_eval, indent=2), encoding="utf-8")

    write_report(output, labeled, all_articles, cv_results, holdout, best_name, cache_eval, external_eval)
    print(cv_results)
    print(f"best_model={best_name}")
    print(f"report={output / 'MODEL_RESULTS.md'}")


if __name__ == "__main__":
    main()
