"""Reusable utilities for the separate educational-text classification demonstration."""

from dataclasses import dataclass
import csv
from pathlib import Path
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


SUPPORTED_CATEGORY_LABELS = (
    "what_it_measures",
    "why_doctors_check_it",
    "common_associations",
    "general_wellness",
    "doctor_questions",
)

REQUIRED_DATASET_COLUMNS = (
    "text",
    "label",
    "group_id",
    "source_note",
    "authorship_status",
    "review_status",
    "review_date",
    "reviewer_role",
)

MIN_MEANINGFUL_TOKENS = 3
VOCABULARY_COVERAGE_WARNING_THRESHOLD = 0.50


@dataclass(frozen=True)
class EducationTextDataset:
    """Validated dataset rows and their report-free provenance metadata."""

    texts: tuple[str, ...]
    labels: tuple[str, ...]
    group_ids: tuple[str, ...]
    source_notes: tuple[str, ...]
    authorship_statuses: tuple[str, ...]
    review_statuses: tuple[str, ...]
    review_dates: tuple[str, ...]
    reviewer_roles: tuple[str, ...]


@dataclass(frozen=True)
class PredictionWarning:
    """A non-clinical warning for invalid or weakly covered input text."""

    code: str
    message: str


@dataclass(frozen=True)
class EducationPrediction:
    """Prediction output without any report-range or clinical classification."""

    text: str
    predicted_category: str
    probabilities: dict[str, float]
    warnings: tuple[PredictionWarning, ...]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_and_validate_dataset(
    dataset_path: str | Path | None = None,
) -> EducationTextDataset:
    """Load and validate the illustrative educational-text CSV."""
    path = (
        _project_root() / "data" / "ml_education_intents.csv"
        if dataset_path is None
        else Path(dataset_path)
    )
    if not path.is_file():
        raise FileNotFoundError(f"Educational-text dataset not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as dataset_file:
        reader = csv.DictReader(dataset_file)
        fieldnames = tuple(reader.fieldnames or ())
        missing_columns = [
            column for column in REQUIRED_DATASET_COLUMNS if column not in fieldnames
        ]
        if missing_columns:
            raise ValueError(
                "Educational-text dataset is missing required columns: "
                + ", ".join(missing_columns)
            )
        rows = list(reader)

    if not rows:
        raise ValueError("Educational-text dataset must contain at least one example.")

    texts: list[str] = []
    labels: list[str] = []
    group_ids: list[str] = []
    source_notes: list[str] = []
    authorship_statuses: list[str] = []
    review_statuses: list[str] = []
    review_dates: list[str] = []
    reviewer_roles: list[str] = []

    seen_texts: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        text = (row.get("text") or "").strip()
        label = (row.get("label") or "").strip()
        group_id = (row.get("group_id") or "").strip()
        source_note = (row.get("source_note") or "").strip()
        authorship_status = (row.get("authorship_status") or "").strip()
        review_status = (row.get("review_status") or "").strip()
        review_date = (row.get("review_date") or "").strip()
        reviewer_role = (row.get("reviewer_role") or "").strip()

        if not text:
            raise ValueError(f"Dataset row {row_number} has empty text.")
        normalized_text = re.sub(r"\s+", " ", text).casefold()
        if normalized_text in seen_texts:
            raise ValueError(f"Dataset row {row_number} duplicates existing text.")
        seen_texts.add(normalized_text)

        if label not in SUPPORTED_CATEGORY_LABELS:
            raise ValueError(
                f"Dataset row {row_number} has unsupported label {label!r}."
            )
        if not group_id:
            raise ValueError(f"Dataset row {row_number} has a missing group_id.")
        if not source_note:
            raise ValueError(f"Dataset row {row_number} has a missing source_note.")
        if authorship_status != "generated":
            raise ValueError(
                f"Dataset row {row_number} must remain authorship_status='generated'."
            )
        if review_status != "unreviewed":
            raise ValueError(
                f"Dataset row {row_number} must remain review_status='unreviewed'."
            )
        if review_date:
            raise ValueError(
                f"Dataset row {row_number} must retain a blank review_date."
            )
        if reviewer_role != "none":
            raise ValueError(
                f"Dataset row {row_number} must retain reviewer_role='none'."
            )

        texts.append(text)
        labels.append(label)
        group_ids.append(group_id)
        source_notes.append(source_note)
        authorship_statuses.append(authorship_status)
        review_statuses.append(review_status)
        review_dates.append(review_date)
        reviewer_roles.append(reviewer_role)

    observed_labels = set(labels)
    missing_labels = set(SUPPORTED_CATEGORY_LABELS) - observed_labels
    if missing_labels:
        raise ValueError(
            "Educational-text dataset is missing labels: "
            + ", ".join(sorted(missing_labels))
        )

    return EducationTextDataset(
        texts=tuple(texts),
        labels=tuple(labels),
        group_ids=tuple(group_ids),
        source_notes=tuple(source_notes),
        authorship_statuses=tuple(authorship_statuses),
        review_statuses=tuple(review_statuses),
        review_dates=tuple(review_dates),
        reviewer_roles=tuple(reviewer_roles),
    )


def build_education_text_pipeline() -> Pipeline:
    """Construct an unfitted TF-IDF and Logistic Regression pipeline."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    sublinear_tf=True,
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=1.0,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def get_pipeline_fit_warning(pipeline: Pipeline) -> PredictionWarning | None:
    """Identify completely unfitted or partially fitted educational pipelines."""
    if not isinstance(pipeline, Pipeline) or not hasattr(pipeline, "named_steps"):
        return PredictionWarning(
            "invalid_pipeline",
            "The supplied pipeline is invalid.",
        )

    vectorizer = pipeline.named_steps.get("tfidf")
    classifier = pipeline.named_steps.get("classifier")
    if not isinstance(vectorizer, TfidfVectorizer):
        return PredictionWarning(
            "invalid_pipeline",
            "The supplied pipeline does not contain a TF-IDF vectorizer.",
        )
    if not isinstance(classifier, LogisticRegression):
        return PredictionWarning(
            "invalid_pipeline",
            "The supplied pipeline does not contain the expected Logistic Regression classifier.",
        )

    vectorizer_is_fitted = hasattr(vectorizer, "vocabulary_")
    classifier_is_fitted = (
        hasattr(classifier, "classes_")
        and hasattr(classifier, "coef_")
    )
    if not vectorizer_is_fitted and not classifier_is_fitted:
        return PredictionWarning(
            "pipeline_unfitted",
            "The educational-text pipeline is not fitted.",
        )
    if vectorizer_is_fitted and not classifier_is_fitted:
        return PredictionWarning(
            "tfidf_fitted_classifier_unfitted",
            "The TF-IDF component is fitted, but the Logistic Regression classifier is not fitted.",
        )
    if not vectorizer_is_fitted and classifier_is_fitted:
        return PredictionWarning(
            "classifier_fitted_vectorizer_unfitted",
            "The classifier is fitted, but the TF-IDF vectorizer is not fitted.",
        )
    if not hasattr(pipeline, "predict") or not hasattr(pipeline, "predict_proba"):
        return PredictionWarning(
            "invalid_pipeline",
            "The fitted pipeline cannot produce predictions and probabilities.",
        )
    return None


def validate_prediction_input(text: object) -> tuple[str, tuple[PredictionWarning, ...]]:
    """Validate empty and short input before running a fitted pipeline."""
    if not isinstance(text, str):
        return "", (
            PredictionWarning(
                "invalid_input",
                "Input must be text.",
            ),
        )

    stripped_text = text.strip()
    if not stripped_text:
        return "", (
            PredictionWarning(
                "empty_input",
                "Enter an educational sentence before requesting a prediction.",
            ),
        )

    meaningful_tokens = re.findall(r"[A-Za-z][A-Za-z'-]*", stripped_text)
    if len(meaningful_tokens) < MIN_MEANINGFUL_TOKENS:
        return stripped_text, (
            PredictionWarning(
                "short_input",
                "Input is short; the category prediction may be unreliable.",
            ),
        )

    return stripped_text, ()


def get_vocabulary_coverage_warnings(
    pipeline: Pipeline,
    text: str,
) -> tuple[PredictionWarning, ...]:
    """Warn when a high proportion of meaningful tokens is outside the fitted TF-IDF vocabulary."""
    fit_warning = get_pipeline_fit_warning(pipeline)
    if fit_warning is not None:
        return (fit_warning,)

    vectorizer = pipeline.named_steps.get("tfidf")
    if not isinstance(vectorizer, TfidfVectorizer) or not hasattr(vectorizer, "vocabulary_"):
        return (
            PredictionWarning(
                "invalid_pipeline",
                "The supplied TF-IDF vectorizer is missing or not fitted.",
            ),
        )

    tokens = re.findall(r"[A-Za-z][A-Za-z'-]*", text.casefold())
    if not tokens:
        return ()

    vocabulary = set(vectorizer.vocabulary_)
    analyzer = vectorizer.build_analyzer()
    analyzed_terms = set(analyzer(text.casefold()))
    meaningful_terms = {
        token for token in tokens if token in vocabulary
    }
    fitted_coverage = (
        len(analyzed_terms & vocabulary) / len(analyzed_terms)
        if analyzed_terms
        else 0.0
    )

    # Analyze unigrams separately to avoid n-gram vocabulary terms being
    # mistaken for out-of-vocabulary content.
    unigram_coverage = (
        len(meaningful_terms) / len(set(tokens))
        if set(tokens)
        else 0.0
    )

    if fitted_coverage < VOCABULARY_COVERAGE_WARNING_THRESHOLD or unigram_coverage < VOCABULARY_COVERAGE_WARNING_THRESHOLD:
        return (
            PredictionWarning(
                "mostly_out_of_vocabulary",
                "Most of the input is outside the fitted vocabulary; the category prediction may be unreliable.",
            ),
        )
    return ()


def predict_education_category(
    pipeline: Pipeline,
    text: str,
) -> EducationPrediction:
    """Return a predicted educational category and all five model probability outputs."""
    validated_text, warnings = validate_prediction_input(text)
    if warnings:
        return EducationPrediction(
            text=validated_text,
            predicted_category="Unavailable",
            probabilities={
                category: 0.0 for category in SUPPORTED_CATEGORY_LABELS
            },
            warnings=warnings,
        )

    fit_warning = get_pipeline_fit_warning(pipeline)
    if fit_warning is not None:
        return EducationPrediction(
            text=validated_text,
            predicted_category="Unavailable",
            probabilities={
                category: 0.0 for category in SUPPORTED_CATEGORY_LABELS
            },
            warnings=(fit_warning,),
        )

    vocabulary_warnings = get_vocabulary_coverage_warnings(pipeline, validated_text)
    warnings += vocabulary_warnings

    warning_codes = {warning.code for warning in warnings}
    if "invalid_pipeline" in warning_codes:
        return EducationPrediction(
            text=validated_text,
            predicted_category="Unavailable",
            probabilities={
                category: 0.0 for category in SUPPORTED_CATEGORY_LABELS
            },
            warnings=warnings,
        )

    classifier = pipeline.named_steps["classifier"]

    try:
        predicted = pipeline.predict([validated_text])[0]
    except Exception:
        return EducationPrediction(
            text=validated_text,
            predicted_category="Unavailable",
            probabilities={
                category: 0.0 for category in SUPPORTED_CATEGORY_LABELS
            },
            warnings=warnings
            + (
                PredictionWarning(
                    "prediction_error",
                    "A prediction could not be generated safely.",
                ),
            ),
        )

    try:
        raw_probabilities = pipeline.predict_proba([validated_text])[0]
    except Exception:
        return EducationPrediction(
            text=validated_text,
            predicted_category="Unavailable",
            probabilities={
                category: 0.0 for category in SUPPORTED_CATEGORY_LABELS
            },
            warnings=warnings
            + (
                PredictionWarning(
                    "probability_error",
                    "Class probabilities could not be generated safely.",
                ),
            ),
        )
    if hasattr(classifier, "classes_"):
        classes = tuple(str(category) for category in classifier.classes_)
    else:
        classes = SUPPORTED_CATEGORY_LABELS

    probability_map = {
        category: 0.0 for category in SUPPORTED_CATEGORY_LABELS
    }
    try:
        raw_probability_values = tuple(float(value) for value in raw_probabilities)
    except (TypeError, ValueError):
        return EducationPrediction(
            text=validated_text,
            predicted_category="Unavailable",
            probabilities=probability_map,
            warnings=warnings
            + (
                PredictionWarning(
                    "probability_error",
                    "Class probabilities were returned in an unsupported format.",
                ),
            ),
        )

    if len(classes) != len(raw_probability_values):
        return EducationPrediction(
            text=validated_text,
            predicted_category="Unavailable",
            probabilities=probability_map,
            warnings=warnings
            + (
                PredictionWarning(
                    "probability_error",
                    "Class probabilities did not align with the fitted categories.",
                ),
            ),
        )

    for category, probability in zip(classes, raw_probability_values):
        if category in probability_map:
            probability_map[category] = probability

    predicted_category = str(predicted)
    if predicted_category not in SUPPORTED_CATEGORY_LABELS:
        return EducationPrediction(
            text=validated_text,
            predicted_category="Unavailable",
            probabilities=probability_map,
            warnings=warnings
            + (
                PredictionWarning(
                    "unsupported_prediction",
                    "The fitted model returned a category outside the supported labels.",
                ),
            ),
        )

    return EducationPrediction(
        text=validated_text,
        predicted_category=predicted_category,
        probabilities=probability_map,
        warnings=warnings,
    )
