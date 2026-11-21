"""Compare confirmed blood-test values with the report's own reference ranges."""

from decimal import Decimal, InvalidOperation
import re
from typing import Mapping, Sequence

from modules.blood_test_catalog import (
    ParameterResolution,
    get_test_definition_by_id,
    resolve_test_definition,
)
from modules.parameter_database import units_are_equivalent


_NUMBER_PATTERN = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_MISSING_VALUES = {
    "",
    "unavailable",
    "not available",
    "not provided",
    "not supplied",
    "n/a",
    "na",
    "none",
    "null",
    "-",
}


def _display_value(value: object) -> str:
    """Keep the supplied value for display, using a clear marker when absent."""
    if value is None:
        return "Unavailable"
    text = str(value)
    return text if text.strip() else "Unavailable"


def _is_missing(value: object) -> bool:
    """Recognize blank and explicit missing-value markers for validation."""
    return value is None or str(value).strip().casefold() in _MISSING_VALUES


def _parse_number(value: str) -> Decimal | None:
    """Parse only a plain integer or decimal value without changing its units."""
    if not re.fullmatch(_NUMBER_PATTERN, value.strip()):
        return None
    try:
        return Decimal(value.strip().replace(",", ""))
    except InvalidOperation:
        return None


def _parse_reference_range(
    reference_text: str,
) -> tuple[str, Decimal, Decimal | None, str, bool] | None:
    """Parse a bounded or one-sided range and retain its optional unit."""
    range_text = reference_text.strip()
    number = _NUMBER_PATTERN

    bounded_match = re.fullmatch(
        rf"\s*(?P<low>{number})\s*(?:-|–|—|\bto\b)\s*"
        rf"(?P<high>{number})\s*(?P<unit>.*?)\s*",
        range_text,
        re.IGNORECASE,
    )
    if bounded_match:
        low = _parse_number(bounded_match.group("low"))
        high = _parse_number(bounded_match.group("high"))
        if low is not None and high is not None:
            unit = bounded_match.group("unit").strip().strip("()[]{} \t")
            return "between", low, high, unit, True

    threshold_match = re.fullmatch(
        rf"\s*(?P<operator><=|>=|<|>|below|under|less\s+than|"
        rf"at\s+most|up\s+to|above|over|greater\s+than|at\s+least)\s*"
        rf"(?P<value>{number})\s*(?P<unit>.*?)\s*",
        range_text,
        re.IGNORECASE,
    )
    if not threshold_match:
        return None

    limit = _parse_number(threshold_match.group("value"))
    if limit is None:
        return None

    operator = re.sub(r"\s+", " ", threshold_match.group("operator").casefold())
    unit = threshold_match.group("unit").strip().strip("()[]{} \t")
    if operator in {"<", "below", "under", "less than"}:
        return "upper", limit, None, unit, False
    if operator in {"<=", "at most", "up to"}:
        return "upper", limit, None, unit, True
    if operator in {">", "above", "over", "greater than"}:
        return "lower", limit, None, unit, False
    return "lower", limit, None, unit, True


_DIRECT_REPORT_UNIT_ALIASES = {
    "wbc_count": frozenset({"/ul", "/µl", "/μl", "cells/ul", "cells/µl", "cells/μl"}),
    "platelet_count": frozenset({"/ul", "/µl", "/μl", "cells/ul", "cells/µl", "cells/μl"}),
}


def _normalize_report_unit_alias(unit: str) -> str:
    """Normalize only spacing and letter case for explicit report-unit aliases."""
    normalized = re.sub(r"\s+", "", unit.strip()).casefold()
    return normalized.replace(" ", "")


def _accept_report_unit_alias(
    test_name: str,
    result_unit: str | None,
) -> ParameterResolution:
    """Resolve a unique parameter name and accept only an approved report-unit alias."""
    resolution = resolve_test_definition(test_name, None)
    if resolution.status != "matched" or resolution.parameter is None:
        return resolution
    if not isinstance(result_unit, str) or not result_unit.strip():
        return resolution

    parameter = resolution.parameter
    parameter_id = parameter["id"]
    canonical_unit = parameter["unit"]
    normalized_result = _normalize_report_unit_alias(result_unit)
    accepted_aliases = _DIRECT_REPORT_UNIT_ALIASES.get(parameter_id, frozenset())

    if units_are_equivalent(result_unit, canonical_unit, parameter_id):
        return ParameterResolution(
            "matched",
            parameter,
            resolution.candidates,
        )
    if normalized_result in accepted_aliases:
        return ParameterResolution(
            "matched",
            parameter,
            resolution.candidates,
        )
    return ParameterResolution(
        "unit_mismatch",
        None,
        resolution.candidates,
    )


def _classify_row(
    row: Mapping[str, object],
    resolution: ParameterResolution,
) -> tuple[str, str, str]:
    """Return the category, panel, and a user-facing reason for one row."""
    result_unit_value = row.get("Unit")
    result_unit = _display_value(result_unit_value)

    if resolution.status == "not_found":
        return "Unable to classify", "Unsupported", "Unsupported test name."
    if resolution.status == "ambiguous":
        candidate_names = ", ".join(
            candidate["parameter_name"] for candidate in resolution.candidates
        )
        return (
            "Unable to classify",
            "Ambiguous",
            f"Ambiguous test name or alias; matching parameters: {candidate_names}.",
        )

    parameter = resolution.parameter
    if resolution.status == "unit_mismatch":
        if parameter is None and len(resolution.candidates) == 1:
            parameter = resolution.candidates[0]
        if parameter is None:
            candidate_names = ", ".join(
                candidate["parameter_name"] for candidate in resolution.candidates
            )
            return (
                "Unable to classify",
                "Ambiguous",
                "Supplied unit does not exactly match any candidate parameter "
                f"for the ambiguous name: {candidate_names}.",
            )
        definition = get_test_definition_by_id(parameter["id"])
        panel = definition.panel if definition else parameter["category_id"]
        return (
            "Unable to classify",
            panel,
            f"Result unit '{result_unit}' is incompatible with catalog unit "
            f"'{parameter['unit']}' for {parameter['parameter_name']}; "
            "no conversion was attempted.",
        )

    if parameter is None:
        return "Unable to classify", "Unsupported", "Unsupported test name."
    definition = get_test_definition_by_id(parameter["id"])
    if definition is None:
        return "Unable to classify", "Unsupported", "Catalog parameter is unavailable."

    reference_value = row.get("Reference range")
    reference_text = _display_value(reference_value)
    result_value = row.get("Result value")
    result_text = _display_value(result_value)
    result = _parse_number(result_text)

    missing_details = []
    if _is_missing(reference_value):
        missing_details.append("Missing laboratory reference range.")
    if _is_missing(result_value) or result is None:
        missing_details.append("Result value is missing or is not a plain number.")
    if _is_missing(result_unit_value):
        missing_details.append("Result unit is unavailable.")
    if missing_details:
        return (
            "Unable to classify",
            definition.panel,
            " ".join(missing_details),
        )
    assert result is not None

    range_types = {
        reference_range["range_type"]
        for reference_range in parameter["typical_reference_ranges"]
    }
    if range_types != {"reference_interval"}:
        range_type_details = ", ".join(sorted(range_types))
        return (
            "Unable to classify",
            definition.panel,
            "Catalog range type(s) "
            f"{range_type_details} are not exclusively reference_interval; "
            "they cannot be treated as ordinary reference intervals.",
        )

    parsed_range = _parse_reference_range(reference_text)
    if parsed_range is None:
        return (
            "Unable to classify",
            definition.panel,
            "Laboratory reference range format is not supported.",
        )

    range_kind, first_bound, second_bound, range_unit, inclusive = parsed_range
    if second_bound is not None and first_bound > second_bound:
        return (
            "Unable to classify",
            definition.panel,
            "Reference range lower bound exceeds its upper bound.",
        )

    if range_unit:
        if not units_are_equivalent(
            result_unit,
            range_unit,
            parameter["id"],
        ):
            return (
                "Unable to classify",
                definition.panel,
                "Result and reference range units are incompatible; no conversion was attempted.",
            )
        unit_note = "Compared using the units stated in the report."
    else:
        unit_note = (
            "Reference range unit is omitted; the result unit is used as the "
            "implied unit for this test row. No conversion was performed."
        )

    if range_kind == "between":
        if result < first_bound:
            classification = "Low"
        elif result > second_bound:
            classification = "High"
        else:
            classification = "Within reference range"
    elif range_kind == "upper":
        is_within = result <= first_bound if inclusive else result < first_bound
        classification = "Within reference range" if is_within else "High"
    else:
        is_within = result >= first_bound if inclusive else result > first_bound
        classification = "Within reference range" if is_within else "Low"

    return classification, definition.panel, unit_note


def interpret_report_rows(
    report_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, str | None]]:
    """Build a separate interpretation view without changing confirmed rows."""
    interpretation_rows: list[dict[str, str | None]] = []
    for row in report_rows:
        test_name = _display_value(row.get("Test name"))
        result_unit_value = row.get("Unit")
        result_unit = (
            None
            if _is_missing(result_unit_value)
            else _display_value(result_unit_value)
        )
        resolution = resolve_test_definition(test_name, result_unit)
        if resolution.status == "unit_mismatch":
            resolution = _accept_report_unit_alias(test_name, result_unit)
        classification, panel, note = _classify_row(row, resolution)
        parameter = resolution.parameter
        if (
            parameter is None
            and resolution.status == "unit_mismatch"
            and len(resolution.candidates) == 1
        ):
            parameter = resolution.candidates[0]
        definition = (
            get_test_definition_by_id(parameter["id"])
            if parameter is not None
            else None
        )
        if definition:
            normalized_test_name = definition.canonical_name
        elif resolution.status == "ambiguous":
            normalized_test_name = "Ambiguous"
        else:
            normalized_test_name = "Unsupported"
        parameter_id = (
            resolution.parameter["id"]
            if resolution.status == "matched" and resolution.parameter is not None
            else None
        )
        interpretation_rows.append(
            {
                "Test name": test_name,
                "Normalized test name": normalized_test_name,
                "Parameter ID": parameter_id,
                "Resolution status": resolution.status,
                "Result value": _display_value(row.get("Result value")),
                "Unit": _display_value(row.get("Unit")),
                "Reference range": _display_value(row.get("Reference range")),
                "Panel": panel,
                "Classification": classification,
                "Notes": note,
            }
        )
    return interpretation_rows
