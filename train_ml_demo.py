"""Train and evaluate the separate educational-text classification demonstration."""

from datetime import date
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile

import joblib
import sklearn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedGroupKFold

from modules.education_text_classifier import (
    SUPPORTED_CATEGORY_LABELS,
    build_education_text_pipeline,
    load_and_validate_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / "data" / "ml_education_intents.csv"
ARTIFACT_PATH = PROJECT_ROOT / "models" / "education_text_classifier.joblib"
EVALUATION_PATH = PROJECT_ROOT / "data" / "ml_model_evaluation.json"

EXPECTED_HELD_OUT_GROUPS = frozenset({"platelets_family", "sleep_activity"})
EXPECTED_TRAINING_GROUPS = frozenset(
    {
        "balanced_meals",
        "general_health_habits",
        "glucose_family",
        "haemoglobin_family",
        "hba1c_family",
        "hydration_routine",
        "movement_routine",
        "wbc_family",
    }
)


def _category_counts(labels) -> dict[str, int]:
    counts = {category: 0 for category in SUPPORTED_CATEGORY_LABELS}
    for label in labels:
        if label in counts:
            counts[label] += 1
    return counts


def validate_outer_fold(texts, labels, group_ids):
    """Reserve Fold 1 of the fixed outer five-fold grouped split."""
    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )
    folds = list(splitter.split(texts, labels, group_ids))
    if len(folds) != 5:
        raise RuntimeError("The fixed outer split did not generate five folds.")

    training_indices, held_out_indices = folds[0]
    development_count = len(training_indices)
    held_out_count = len(held_out_indices)
    if development_count != 20:
        raise RuntimeError(
            f"Fold 1 development partition must contain exactly 20 rows; "
            f"found {development_count}."
        )
    if held_out_count != 5:
        raise RuntimeError(
            f"Fold 1 held-out partition must contain exactly 5 rows; "
            f"found {held_out_count}."
        )

    development_category_counts = _category_counts(
        [labels[index] for index in training_indices]
    )
    held_out_category_counts = _category_counts(
        [labels[index] for index in held_out_indices]
    )
    for category in SUPPORTED_CATEGORY_LABELS:
        if development_category_counts[category] != 4:
            raise RuntimeError(
                f"Fold 1 development partition must contain exactly 4 rows "
                f"for {category}; found "
                f"{development_category_counts[category]}."
            )
        if held_out_category_counts[category] != 1:
            raise RuntimeError(
                f"Fold 1 held-out partition must contain exactly 1 row "
                f"for {category}; found "
                f"{held_out_category_counts[category]}."
            )

    training_groups = frozenset(group_ids[index] for index in training_indices)
    held_out_groups = frozenset(group_ids[index] for index in held_out_indices)

    if training_groups != EXPECTED_TRAINING_GROUPS:
        raise RuntimeError(
            "Fold 1 training groups differ from the approved split. "
            f"Expected {sorted(EXPECTED_TRAINING_GROUPS)}, "
            f"found {sorted(training_groups)}."
        )
    if held_out_groups != EXPECTED_HELD_OUT_GROUPS:
        raise RuntimeError(
            "Fold 1 held-out groups differ from the approved split. "
            f"Expected {sorted(EXPECTED_HELD_OUT_GROUPS)}, "
            f"found {sorted(held_out_groups)}."
        )
    if training_groups & held_out_groups:
        raise RuntimeError("Fold 1 contains group overlap.")

    training_labels = {labels[index] for index in training_indices}
    held_out_labels = {labels[index] for index in held_out_indices}
    if training_labels != set(SUPPORTED_CATEGORY_LABELS):
        raise RuntimeError("Fold 1 training partition is missing one or more categories.")
    if held_out_labels != set(SUPPORTED_CATEGORY_LABELS):
        raise RuntimeError("Fold 1 held-out partition is missing one or more categories.")

    return training_indices, held_out_indices, training_groups, held_out_groups


def validate_inner_folds(texts, labels, group_ids):
    """Generate and validate four inner grouped folds before pipeline fitting."""
    splitter = StratifiedGroupKFold(
        n_splits=4,
        shuffle=True,
        random_state=42,
    )
    folds = list(splitter.split(texts, labels, group_ids))
    if len(folds) != 4:
        raise RuntimeError("The inner split did not generate four folds.")

    for fold_number, (inner_training, inner_validation) in enumerate(folds, start=1):
        training_groups = frozenset(group_ids[index] for index in inner_training)
        validation_groups = frozenset(group_ids[index] for index in inner_validation)
        overlap = training_groups & validation_groups
        if overlap:
            raise RuntimeError(
                f"Inner fold {fold_number} contains group overlap: "
                f"{sorted(overlap)}"
            )

        training_labels = {labels[index] for index in inner_training}
        validation_labels = {labels[index] for index in inner_validation}
        if training_labels != set(SUPPORTED_CATEGORY_LABELS):
            raise RuntimeError(
                f"Inner fold {fold_number} training partition is missing categories."
            )
        if validation_labels != set(SUPPORTED_CATEGORY_LABELS):
            raise RuntimeError(
                f"Inner fold {fold_number} validation partition is missing categories."
            )

    return folds


def _metric_summary(actual, predicted) -> dict[str, object]:
    precision, recall, f1, support = precision_recall_fscore_support(
        actual,
        predicted,
        labels=list(SUPPORTED_CATEGORY_LABELS),
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        actual,
        predicted,
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        actual,
        predicted,
        average="weighted",
        zero_division=0,
    )

    per_class = {}
    for index, category in enumerate(SUPPORTED_CATEGORY_LABELS):
        per_class[category] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }

    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro": {
            "precision": float(macro_precision),
            "recall": float(macro_recall),
            "f1": float(macro_f1),
        },
        "weighted": {
            "precision": float(weighted_precision),
            "recall": float(weighted_recall),
            "f1": float(weighted_f1),
        },
        "per_class": per_class,
    }


def _write_outputs_safely(
    artifact_payload: dict[str, object],
    evaluation_metadata: dict[str, object],
) -> None:
    """Prepare both temporary outputs before replacing existing final paths."""
    artifact_temp = None
    evaluation_temp = None
    artifact_backup = None
    artifact_replaced = False
    rollback_failed = False
    preserved_backup_path = None

    try:
        ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)

        artifact_fd, artifact_temp = tempfile.mkstemp(
            prefix=".education_text_classifier.",
            suffix=".tmp",
            dir=ARTIFACT_PATH.parent,
        )
        os.close(artifact_fd)
        joblib.dump(artifact_payload, artifact_temp)

        evaluation_fd, evaluation_temp = tempfile.mkstemp(
            prefix=".ml_model_evaluation.",
            suffix=".tmp",
            dir=EVALUATION_PATH.parent,
        )
        os.close(evaluation_fd)
        Path(evaluation_temp).write_text(
            json.dumps(evaluation_metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        if ARTIFACT_PATH.exists():
            backup_fd, artifact_backup = tempfile.mkstemp(
                prefix=".education_text_classifier.backup.",
                suffix=".tmp",
                dir=ARTIFACT_PATH.parent,
            )
            os.close(backup_fd)
            shutil.copy2(ARTIFACT_PATH, artifact_backup)

        try:
            os.replace(artifact_temp, ARTIFACT_PATH)
            artifact_temp = None
            artifact_replaced = True
            os.replace(evaluation_temp, EVALUATION_PATH)
            evaluation_temp = None
        except Exception as replacement_error:
            if artifact_replaced:
                if artifact_backup is not None:
                    try:
                        os.replace(artifact_backup, ARTIFACT_PATH)
                        artifact_backup = None
                    except Exception as rollback_error:
                        rollback_failed = True
                        preserved_backup_path = artifact_backup
                        raise RuntimeError(
                            "Evaluation JSON replacement failed: "
                            f"{replacement_error}. "
                            "Previous model rollback failed: "
                            f"{rollback_error}. "
                            "The recovery backup was preserved at "
                            f"{artifact_backup}."
                        ) from rollback_error
                else:
                    try:
                        if ARTIFACT_PATH.exists():
                            ARTIFACT_PATH.unlink()
                    except Exception as rollback_error:
                        rollback_failed = True
                        raise RuntimeError(
                            "Evaluation JSON replacement failed: "
                            f"{replacement_error}. "
                            "Rollback failed while removing the newly installed "
                            f"model artifact: {rollback_error}. "
                            "No previous model backup existed."
                        ) from rollback_error
            raise RuntimeError(
                "Model and evaluation outputs could not be replaced safely: "
                f"{replacement_error}"
            ) from replacement_error
    except RuntimeError:
        raise
    except Exception as preparation_error:
        raise RuntimeError(
            "Temporary model or evaluation output could not be written: "
            f"{preparation_error}"
        ) from preparation_error
    finally:
        for temporary_path in (
            artifact_temp,
            evaluation_temp,
            artifact_backup,
        ):
            if temporary_path is not None:
                if rollback_failed and temporary_path == preserved_backup_path:
                    continue
                try:
                    Path(temporary_path).unlink(missing_ok=True)
                except Exception as cleanup_error:
                    sys.stderr.write(
                        "Warning: could not remove temporary or backup file "
                        f"{temporary_path}: {cleanup_error}\n"
                    )


def run_inner_model_selection(development_texts, development_labels, inner_folds):
    """Run a simple documented pipeline across only the inner training folds."""
    fold_metrics = []
    for fold_number, (inner_training, inner_validation) in enumerate(inner_folds, start=1):
        pipeline = build_education_text_pipeline()
        pipeline.fit(
            [development_texts[index] for index in inner_training],
            [development_labels[index] for index in inner_training],
        )
        validation_predictions = pipeline.predict(
            [development_texts[index] for index in inner_validation]
        )
        validation_actual = [
            development_labels[index] for index in inner_validation
        ]
        metrics = _metric_summary(validation_actual, validation_predictions)
        fold_metrics.append(
            {
                "fold": fold_number,
                "accuracy": metrics["accuracy"],
                "macro": metrics["macro"],
                "weighted": metrics["weighted"],
            }
        )

    if not fold_metrics:
        raise RuntimeError("Inner grouped validation produced no fold metrics.")

    mean_accuracy = sum(item["accuracy"] for item in fold_metrics) / len(fold_metrics)
    return {
        "configuration": {
            "tfidf": {
                "lowercase": True,
                "strip_accents": "unicode",
                "ngram_range": [1, 2],
                "sublinear_tf": True,
                "min_df": 1,
                "max_df": 1.0,
            },
            "logistic_regression": {
                "max_iter": 1000,
                "random_state": 42,
                "class_weight": "balanced",
                "multi_class": "not_explicitly_set",
            },
        },
        "fold_metrics": fold_metrics,
        "mean_accuracy": float(mean_accuracy),
    }


def train_and_evaluate():
    """Run fixed grouped validation and one exploratory outer-held-out evaluation."""
    dataset = load_and_validate_dataset(DATASET_PATH)
    texts = list(dataset.texts)
    labels = list(dataset.labels)
    group_ids = list(dataset.group_ids)

    training_indices, held_out_indices, training_groups, held_out_groups = (
        validate_outer_fold(texts, labels, group_ids)
    )

    development_texts = [texts[index] for index in training_indices]
    development_labels = [labels[index] for index in training_indices]
    development_group_ids = [group_ids[index] for index in training_indices]

    inner_folds = validate_inner_folds(
        development_texts,
        development_labels,
        development_group_ids,
    )
    inner_selection = run_inner_model_selection(
        development_texts,
        development_labels,
        inner_folds,
    )

    final_pipeline = build_education_text_pipeline()
    final_pipeline.fit(development_texts, development_labels)

    held_out_texts = [texts[index] for index in held_out_indices]
    held_out_labels = [labels[index] for index in held_out_indices]
    held_out_predictions = final_pipeline.predict(held_out_texts)
    held_out_metrics = _metric_summary(held_out_labels, held_out_predictions)
    held_out_metrics["confusion_matrix"] = confusion_matrix(
        held_out_labels,
        held_out_predictions,
        labels=list(SUPPORTED_CATEGORY_LABELS),
    ).tolist()
    held_out_metrics["confusion_matrix_labels"] = list(SUPPORTED_CATEGORY_LABELS)

    artifact_payload = {
        "pipeline": final_pipeline,
        "categories": list(SUPPORTED_CATEGORY_LABELS),
        "unknown_policy": {
            "probability_threshold": "not_used",
            "empty_input": "reject",
            "minimum_meaningful_tokens": 3,
            "vocabulary_coverage": "advisory_warning_only",
        },
        "dataset_status": {
            "authorship_status": "generated",
            "review_status": "unreviewed",
            "evaluation_warning": (
                "Exploratory results from a tiny generated and unreviewed dataset."
            ),
        },
        "split_metadata": {
            "outer": {
                "splitter": "StratifiedGroupKFold",
                "n_splits": 5,
                "shuffle": True,
                "random_state": 42,
                "held_out_fold": 1,
                "training_group_ids": sorted(training_groups),
                "held_out_group_ids": sorted(held_out_groups),
            },
            "inner": {
                "splitter": "StratifiedGroupKFold",
                "n_splits": 4,
                "shuffle": True,
                "random_state": 42,
            },
        },
        "training_metadata": {
            "training_row_count": len(development_texts),
            "training_category_counts": _category_counts(development_labels),
            "inner_selection": inner_selection,
        },
    }

    evaluation_metadata = {
        "dataset": {
            "path": "data/ml_education_intents.csv",
            "total_rows": len(texts),
            "category_counts": _category_counts(labels),
            "authorship_status": "generated",
            "review_status": "unreviewed",
            "review_date": "",
            "reviewer_role": "none",
            "source_notes": "generated synthetic educational examples; no external source claim",
        },
        "split": artifact_payload["split_metadata"],
        "partition_counts": {
            "development_rows": len(development_texts),
            "development_category_counts": _category_counts(development_labels),
            "held_out_rows": len(held_out_texts),
            "held_out_category_counts": _category_counts(held_out_labels),
        },
        "inner_model_selection": inner_selection,
        "held_out_evaluation": {
            "label": "Exploratory evaluation from a tiny generated, unreviewed dataset",
            **held_out_metrics,
        },
        "package_versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "evaluation_date": date.today().isoformat(),
    }
    _write_outputs_safely(artifact_payload, evaluation_metadata)

    return held_out_metrics, ARTIFACT_PATH, EVALUATION_PATH


def main() -> int:
    """Explicit terminal entry point; importing this script does not train."""
    try:
        metrics, artifact_path, evaluation_path = train_and_evaluate()
    except Exception as error:
        print(f"Training stopped: {error}", file=sys.stderr)
        return 1

    print("Exploratory held-out evaluation completed once.")
    print(f"Accuracy: {metrics['accuracy']:.6f}")
    print(f"Artifact saved: {artifact_path}")
    print(f"Evaluation metadata saved: {evaluation_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
