"""Build deterministic educational explanations for interpreted blood-test rows."""

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_EDUCATION_PATH = _PROJECT_ROOT / "data" / "parameter_education.json"
_LANGUAGES = ("English", "Hinglish")
_CONTENT_FIELDS = (
    "what_it_measures",
    "why_doctors_check_it",
    "general_wellness",
    "doctor_questions",
)
_ASSOCIATION_DIRECTIONS = {"high", "low", "general"}


class _InvalidEducationCatalogError(ValueError):
    """Raised when the education JSON is unreadable or has an invalid schema."""


def _require_text(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _InvalidEducationCatalogError(
            f"{location} must be a non-empty string."
        )
    return value


def _validate_catalog(data: Any) -> None:
    if not isinstance(data, dict) or not isinstance(data.get("parameters"), list):
        raise _InvalidEducationCatalogError(
            "The parameter education database must be an object with a "
            "'parameters' list."
        )

    seen_ids: set[str] = set()
    for index, record in enumerate(data["parameters"]):
        location = f"parameters[{index}]"
        if not isinstance(record, dict):
            raise _InvalidEducationCatalogError(f"{location} must be an object.")

        parameter_id = _require_text(
            record.get("parameter_id"),
            f"{location}.parameter_id",
        )
        if parameter_id in seen_ids:
            raise _InvalidEducationCatalogError(
                f"Duplicate parameter_id {parameter_id!r}."
            )
        seen_ids.add(parameter_id)
        _require_text(record.get("parameter_name"), f"{location}.parameter_name")
        _require_text(
            record.get("content_review_status"),
            f"{location}.content_review_status",
        )

        for field in _CONTENT_FIELDS:
            localized = record.get(field)
            if not isinstance(localized, dict):
                raise _InvalidEducationCatalogError(
                    f"{location}.{field} must be an object."
                )
            for language in _LANGUAGES:
                value = localized.get(language)
                language_location = f"{location}.{field}.{language}"
                if field in {"common_associations", "doctor_questions"}:
                    if not isinstance(value, list) or not all(
                        isinstance(item, str) and item.strip() for item in value
                    ):
                        raise _InvalidEducationCatalogError(
                            f"{language_location} must be a list of non-empty strings."
                        )
                else:
                    _require_text(value, language_location)

        associations = record.get("common_associations")
        if not isinstance(associations, list):
            raise _InvalidEducationCatalogError(
                f"{location}.common_associations must be a list."
            )
        for association_index, association in enumerate(associations):
            association_location = (
                f"{location}.common_associations[{association_index}]"
            )
            if not isinstance(association, dict):
                raise _InvalidEducationCatalogError(
                    f"{association_location} must be an object."
                )
            direction = association.get("direction")
            if (
                not isinstance(direction, str)
                or direction not in _ASSOCIATION_DIRECTIONS
            ):
                allowed_directions = ", ".join(sorted(_ASSOCIATION_DIRECTIONS))
                raise _InvalidEducationCatalogError(
                    f"{association_location}.direction must be one of: "
                    f"{allowed_directions}."
                )
            for language in _LANGUAGES:
                _require_text(
                    association.get(language),
                    f"{association_location}.{language}",
                )

        sources = record.get("sources")
        if not isinstance(sources, list):
            raise _InvalidEducationCatalogError(
                f"{location}.sources must be a list."
            )
        for source_index, source in enumerate(sources):
            source_location = f"{location}.sources[{source_index}]"
            if not isinstance(source, dict):
                raise _InvalidEducationCatalogError(
                    f"{source_location} must be an object."
                )
            _require_text(source.get("title"), f"{source_location}.title")
            _require_text(source.get("url"), f"{source_location}.url")


def _load_catalog() -> dict[str, Any]:
    try:
        with _EDUCATION_PATH.open(encoding="utf-8") as education_file:
            data = json.load(education_file)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Parameter education database was not found at {_EDUCATION_PATH}."
        ) from error
    except json.JSONDecodeError as error:
        raise _InvalidEducationCatalogError(
            "Parameter education database contains invalid JSON at "
            f"line {error.lineno}, column {error.colno}: {error.msg}."
        ) from error
    except UnicodeDecodeError as error:
        raise _InvalidEducationCatalogError(
            "Parameter education database contains invalid UTF-8 text."
        ) from error

    _validate_catalog(data)
    return data


def load_education_catalog() -> dict[str, Any]:
    """Return a deep copy of the validated educational content database."""
    return deepcopy(_load_catalog())


def _localized_message(language: str, english: str, hinglish: str) -> str:
    return hinglish if language == "Hinglish" else english


def _safe_fallback(
    *,
    parameter_id: str | None,
    resolution_status: Any,
    classification: Any,
    language: str,
    interpreter_reason: Any,
    content_review_status: str,
    fallback_message: str,
) -> dict[str, Any]:
    return {
        "parameter_id": parameter_id,
        "parameter_name": None,
        "resolution_status": resolution_status,
        "classification": classification,
        "language": language,
        "what_it_measures": None,
        "why_doctors_check_it": None,
        "relevant_associations": [],
        "general_wellness": None,
        "doctor_questions": [],
        "content_review_status": content_review_status,
        "content_is_reviewed": False,
        "content_notice": _localized_message(
            language,
            "Educational content is unavailable or not applicable to this row.",
            "Is row ke liye educational content uplabdh nahi hai ya lagu nahi hota.",
        ),
        "sources": [],
        "content_error": None,
        "interpreter_reason": interpreter_reason,
        "fallback_message": fallback_message,
    }


def _relevant_associations(
    associations: list[dict[str, str]],
    classification: str,
    language: str,
) -> list[str]:
    if classification in {"High", "Low"}:
        directions = {classification.casefold(), "general"}
    elif classification == "Within reference range":
        directions = {"general"}
    else:
        return []

    return [
        association[language]
        for association in associations
        if association["direction"] in directions
    ]


def build_educational_explanation(
    interpretation_row: Mapping[str, Any],
    language: str = "English",
) -> dict[str, Any]:
    """Combine one interpreter row with educational content without classifying it."""
    if not isinstance(interpretation_row, Mapping):
        raise TypeError("interpretation_row must be a mapping.")

    requested_language = language if isinstance(language, str) else ""
    language_key = requested_language.casefold()
    language_used = (
        "Hinglish" if language_key == "hinglish" else "English"
    )
    language_fallback = language_key not in {"english", "hinglish"}

    parameter_id_value = interpretation_row.get("Parameter ID")
    parameter_id = (
        parameter_id_value.strip()
        if isinstance(parameter_id_value, str) and parameter_id_value.strip()
        else None
    )
    resolution_status = interpretation_row.get("Resolution status")
    classification = interpretation_row.get("Classification")
    interpreter_reason = (
        interpretation_row.get("Notes")
        if classification == "Unable to classify"
        else None
    )
    missing_content_message = _localized_message(
        language_used,
        "No reviewed educational explanation is available for this parameter.",
        "Is parameter ke liye reviewed educational explanation uplabdh nahi hai.",
    )

    if parameter_id is None or resolution_status != "matched":
        result = _safe_fallback(
            parameter_id=parameter_id,
            resolution_status=resolution_status,
            classification=classification,
            language=language_used,
            interpreter_reason=interpreter_reason,
            content_review_status="not_loaded",
            fallback_message=_localized_message(
                language_used,
                "The test identity or unit was not fully resolved, so parameter-specific education is not shown.",
                "Test ki pehchaan ya unit poori tarah resolve nahi hui, isliye parameter-specific jaankari nahi dikhayi gayi.",
            ),
        )
        result["language_fallback"] = language_fallback
        return result

    try:
        catalog = load_education_catalog()
    except FileNotFoundError:
        result = _safe_fallback(
            parameter_id=parameter_id,
            resolution_status=resolution_status,
            classification=classification,
            language=language_used,
            interpreter_reason=interpreter_reason,
            content_review_status="unavailable",
            fallback_message=_localized_message(
                language_used,
                "The educational content file is unavailable; no parameter-specific explanation is shown.",
                "Educational content file uplabdh nahi hai; parameter-specific explanation nahi dikhayi gayi.",
            ),
        )
        result["language_fallback"] = language_fallback
        return result
    except _InvalidEducationCatalogError as error:
        result = _safe_fallback(
            parameter_id=parameter_id,
            resolution_status=resolution_status,
            classification=classification,
            language=language_used,
            interpreter_reason=interpreter_reason,
            content_review_status="invalid_content",
            fallback_message=_localized_message(
                language_used,
                "The educational content is invalid, so no parameter-specific explanation is shown.",
                "Educational content valid nahi hai, isliye parameter-specific explanation nahi dikhayi gayi.",
            ),
        )
        result["content_error"] = str(error)
        result["language_fallback"] = language_fallback
        return result

    record = next(
        (
            item
            for item in catalog["parameters"]
            if item["parameter_id"] == parameter_id
        ),
        None,
    )
    if record is None:
        result = _safe_fallback(
            parameter_id=parameter_id,
            resolution_status=resolution_status,
            classification=classification,
            language=language_used,
            interpreter_reason=interpreter_reason,
            content_review_status="missing",
            fallback_message=missing_content_message,
        )
        result["language_fallback"] = language_fallback
        return result

    review_status = record["content_review_status"]
    content_is_reviewed = review_status == "reviewed"
    content_notice = _localized_message(
        language_used,
        (
            "Educational content has been reviewed, but it does not diagnose a condition "
            "or change the report's classification."
            if content_is_reviewed
            else "This educational content is pending review and is not a clinical assessment."
        ),
        (
            "Educational content review ho chuka hai, lekin yeh diagnosis nahi karta "
            "aur report ki classification nahi badalta."
            if content_is_reviewed
            else "Yeh educational content review ke liye pending hai aur clinical assessment nahi hai."
        ),
    )

    relevant_associations: list[str] = []
    if content_is_reviewed and classification in {"High", "Low"}:
        relevant_associations = _relevant_associations(
            record["common_associations"],
            classification,
            language_used,
        )
    elif content_is_reviewed and classification == "Within reference range":
        relevant_associations = _relevant_associations(
            record["common_associations"],
            classification,
            language_used,
        )

    wellness = None
    if (
        content_is_reviewed
        and classification in {"High", "Low", "Within reference range"}
    ):
        wellness = record["general_wellness"][language_used]

    return {
        "parameter_id": parameter_id,
        "parameter_name": record["parameter_name"],
        "resolution_status": resolution_status,
        "classification": classification,
        "language": language_used,
        "what_it_measures": record["what_it_measures"][language_used],
        "why_doctors_check_it": record["why_doctors_check_it"][language_used],
        "relevant_associations": relevant_associations,
        "general_wellness": wellness,
        "doctor_questions": deepcopy(record["doctor_questions"][language_used]),
        "content_review_status": review_status,
        "content_is_reviewed": content_is_reviewed,
        "content_notice": content_notice,
        "sources": deepcopy(record["sources"]),
        "content_error": None,
        "interpreter_reason": interpreter_reason,
        "fallback_message": (
            _localized_message(
                language_used,
                "Possible associations and wellness information are withheld until the content is reviewed.",
                "Mumkin sambandh aur wellness jaankari content review hone tak nahi dikhayi jaati.",
            )
            if not content_is_reviewed
            else None
        ),
        "language_fallback": language_fallback,
    }
