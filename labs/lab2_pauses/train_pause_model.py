"""Train pause-presence and pause-duration models from RUSLAN boundaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit

from pause_model import BoundaryFeatureBuilder


def sklearn_compatible_sparse(matrix):
    """Return CSR with 32-bit structural indices required by scikit-learn."""

    result = matrix.tocsr()
    result.indices = result.indices.astype(np.int32, copy=False)
    result.indptr = result.indptr.astype(np.int32, copy=False)
    return result


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def read_rows(path: Path, min_confidence: float) -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = []
    with path.open("r", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            if min(float(row["left_confidence"]), float(row["right_confidence"])) < min_confidence:
                continue
            rows.append(
                row
                | {
                    "position_ratio": float(row["position_ratio"]),
                    "gap_ms": float(row["gap_ms"]),
                    "has_pause": int(row["has_pause"]),
                }
            )
    return rows


def train(
    input_path: Path,
    output_path: Path,
    metrics_path: Path,
    min_confidence: float = 0.5,
    test_size: float = 0.2,
    random_state: int = 2026,
) -> dict[str, object]:
    rows = read_rows(input_path, min_confidence)
    if len(rows) < 100:
        raise ValueError(f"not enough reliable boundaries to train: {len(rows)}")

    groups = np.asarray([str(row["utterance_id"]) for row in rows])
    labels = np.asarray([int(row["has_pause"]) for row in rows], dtype=np.int8)
    outer_splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_indices, test_indices = next(outer_splitter.split(rows, labels, groups))
    inner_splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state + 1)
    fit_relative, validation_relative = next(
        inner_splitter.split(
            train_indices,
            labels[train_indices],
            groups[train_indices],
        )
    )
    fit_indices = train_indices[fit_relative]
    validation_indices = train_indices[validation_relative]

    feature_builder = BoundaryFeatureBuilder()
    feature_dicts = feature_builder.transform(rows)
    evaluation_vectorizer = DictVectorizer(sparse=True)
    fit_matrix = sklearn_compatible_sparse(
        evaluation_vectorizer.fit_transform(
            [feature_dicts[index] for index in fit_indices]
        )
    )
    validation_matrix = sklearn_compatible_sparse(
        evaluation_vectorizer.transform(
            [feature_dicts[index] for index in validation_indices]
        )
    )
    test_matrix = sklearn_compatible_sparse(
        evaluation_vectorizer.transform(
            [feature_dicts[index] for index in test_indices]
        )
    )
    y_fit = labels[fit_indices]
    y_validation = labels[validation_indices]
    y_test = labels[test_indices]

    evaluation_classifier = LogisticRegression(
        class_weight="balanced",
        solver="liblinear",
        max_iter=1_000,
        random_state=random_state,
    )
    evaluation_classifier.fit(fit_matrix, y_fit)
    validation_probabilities = evaluation_classifier.predict_proba(validation_matrix)[:, 1]
    thresholds = np.linspace(0.10, 0.90, 81)
    threshold_scores = [
        f1_score(y_validation, validation_probabilities >= threshold)
        for threshold in thresholds
    ]
    best_threshold = float(thresholds[int(np.argmax(threshold_scores))])
    test_probabilities = evaluation_classifier.predict_proba(test_matrix)[:, 1]
    predictions = (test_probabilities >= best_threshold).astype(np.int8)

    baseline_predictions = np.asarray(
        [rows[index]["punctuation"] != "NONE" for index in test_indices], dtype=np.int8
    )

    positive_fit_mask = y_fit == 1
    if positive_fit_mask.sum() < 10:
        raise ValueError("not enough positive pause examples for duration regression")
    fit_duration_values = np.asarray(
        [float(rows[index]["gap_ms"]) for index in fit_indices]
    )[positive_fit_mask]
    duration_targets = np.log1p(fit_duration_values)
    evaluation_regressor = Ridge(alpha=10.0, solver="lsqr")
    evaluation_regressor.fit(fit_matrix[positive_fit_mask], duration_targets)

    positive_test_mask = y_test == 1
    duration_true = np.asarray(
        [float(rows[index]["gap_ms"]) for index in test_indices]
    )[positive_test_mask]
    duration_predicted = np.maximum(
        0.0, np.expm1(evaluation_regressor.predict(test_matrix[positive_test_mask]))
    )
    duration_baseline = np.full_like(duration_true, np.median(fit_duration_values))

    metrics: dict[str, object] = {
        "input_boundaries": len(rows),
        "input_utterances": len(set(groups)),
        "train_boundaries": int(len(train_indices)),
        "fit_boundaries": int(len(fit_indices)),
        "validation_boundaries": int(len(validation_indices)),
        "test_boundaries": int(len(test_indices)),
        "train_utterances": int(len(set(groups[train_indices]))),
        "test_utterances": int(len(set(groups[test_indices]))),
        "positive_share": float(labels.mean()),
        "min_alignment_confidence": min_confidence,
        "classification_threshold": best_threshold,
        "baseline": classification_metrics(y_test, baseline_predictions),
        "classifier": classification_metrics(y_test, predictions),
        "duration_baseline_mae_ms": float(
            mean_absolute_error(duration_true, duration_baseline)
        ),
        "duration_mae_ms": float(mean_absolute_error(duration_true, duration_predicted)),
        "duration_test_examples": int(positive_test_mask.sum()),
    }

    # Fit the artifact used by the service on every available boundary.  The
    # reported metrics above remain untouched estimates from the held-out test.
    vectorizer = DictVectorizer(sparse=True)
    full_matrix = sklearn_compatible_sparse(vectorizer.fit_transform(feature_dicts))
    classifier = LogisticRegression(
        class_weight="balanced",
        solver="liblinear",
        max_iter=1_000,
        random_state=random_state,
    )
    classifier.fit(full_matrix, labels)
    positive_full_mask = labels == 1
    regressor = Ridge(alpha=10.0, solver="lsqr")
    regressor.fit(
        full_matrix[positive_full_mask],
        np.log1p(np.asarray([float(row["gap_ms"]) for row in rows])[positive_full_mask]),
    )

    bundle = {
        "vectorizer": vectorizer,
        "classifier": classifier,
        "regressor": regressor,
        "classification_threshold": best_threshold,
        "metrics": metrics,
        "feature_names": list(vectorizer.get_feature_names_out()),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def build_argument_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=project_root / "data" / "lab2_pauses" / "pause_boundaries.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "data" / "lab2_pauses" / "pause_model.joblib",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=project_root / "data" / "lab2_pauses" / "model_metrics.json",
    )
    parser.add_argument("--min-confidence", type=float, default=0.5)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    metrics = train(args.input, args.output, args.metrics, min_confidence=args.min_confidence)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
