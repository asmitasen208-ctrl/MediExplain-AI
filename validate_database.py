"""Validate the structure and reference-range formats of the blood-test JSON."""

import json
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from urllib.parse import urlparse


DATABASE_PATH = Path(__file__).resolve().parent / "data" / "blood_test_parameters.json"
EXPECTED_PARAMETER_COUNT = 87
NUMBER_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
UNIT_PATTERN = re.compile(r"^[A-Za-z0-9µμ%°/().^+\-\s]+$")
RANGE_PATTERN = re.compile(
    rf"^\s*(?P<low>{NUMBER_PATTERN})\s*(?:-|–|—|\bto\b)\s*"
    rf"(?P<high>{NUMBER_PATTERN})\s*$",
    re.IGNORECASE,
)
THRESHOLD_PATTERN = re.compile(
    rf"^\s*(?P<operator><=|>=|<|>|≤|≥)\s*(?P<value>{NUMBER_PATTERN})\s*$"
)
SINGLE_VALUE_PATTERN = re.compile(rf"^\s*{NUMBER_PATTERN}\s*$")

REQUIRED_PARAMETER_FIELDS = {
    "id",
    "parameter_name",
    "full_name",
    "aliases",
    "unit",
    "category_id",
    "typical_reference_ranges",
    "description",
}
REQUIRED_RANGE_FIELDS = {
    "population",
    "range",
    "range_type",
    "qualifications",
}
REQUIRED_SOURCE_FIELDS = {
    "id",
    "title",
    "organization",
    "url",
    "used_for",
}
NULL_RANGE_TYPES = {
    "assay_specific",
    "no_single_reference_interval",
    "risk_based_target",
}
KNOWN_RANGE_TYPES = NULL_RANGE_TYPES | {
    "reference_interval",
    "clinical_decision_threshold",
    "general_reference_threshold",
    "nutrient_adequacy_reference",
}


def normalize_label(value: str) -> str:
    """Normalize names for duplicate and alias-collision checks."""
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def parse_range(value: str) -> tuple[str, Decimal | None, Decimal | None] | None:
    """Recognize bounded, threshold, and single-number range formats."""
    bounded = RANGE_PATTERN.fullmatch(value)
    if bounded:
        try:
            low = Decimal(bounded.group("low"))
            high = Decimal(bounded.group("high"))
        except InvalidOperation:
            return None
        if low > high:
            return None
        return "bounded", low, high

    threshold = THRESHOLD_PATTERN.fullmatch(value)
    if threshold:
        try:
            limit = Decimal(threshold.group("value"))
        except InvalidOperation:
            return None
        return "threshold", limit, None

    if SINGLE_VALUE_PATTERN.fullmatch(value):
        try:
            point = Decimal(value.strip())
        except InvalidOperation:
            return None
        return "single_value", point, point

    return None


def add_message(messages: list[str], location: str, message: str) -> None:
    messages.append(f"{location}: {message}")


def validate_database(data: object) -> tuple[list[str], list[str], dict[str, int]]:
    """Return errors, warnings, and summary counts for a decoded JSON value."""
    errors: list[str] = []
    warnings: list[str] = []
    counts = {
        "parameters": 0,
        "categories": 0,
        "sources": 0,
        "ranges": 0,
        "bounded_ranges": 0,
        "threshold_ranges": 0,
        "single_value_ranges": 0,
        "null_ranges": 0,
        "qualified_ranges": 0,
        "units": 0,
    }

    if not isinstance(data, dict):
        return ["root: expected a JSON object"], warnings, counts

    categories = data.get("categories")
    if not isinstance(categories, list):
        add_message(errors, "categories", "expected a list")
        categories = []

    category_ids: set[str] = set()
    for index, category in enumerate(categories):
        location = f"categories[{index}]"
        if not isinstance(category, dict):
            add_message(errors, location, "expected an object")
            continue
        category_id = category.get("id")
        if not isinstance(category_id, str) or not category_id.strip():
            add_message(errors, location, "missing or invalid category id")
            continue
        if category_id in category_ids:
            add_message(errors, location, f"duplicate category id {category_id!r}")
        category_ids.add(category_id)
        for field in ("name", "description"):
            if not isinstance(category.get(field), str) or not category[field].strip():
                add_message(errors, location, f"missing or invalid {field}")
    counts["categories"] = len(category_ids)

    parameters = data.get("parameters")
    if not isinstance(parameters, list):
        add_message(errors, "parameters", "expected a list")
        parameters = []
    counts["parameters"] = len(parameters)

    parameter_ids: dict[str, int] = {}
    parameter_names: dict[str, int] = {}
    alias_owners: dict[str, tuple[str, int, str]] = {}
    for index, parameter in enumerate(parameters):
        location = f"parameters[{index}]"
        if not isinstance(parameter, dict):
            add_message(errors, location, "expected an object")
            continue

        missing_fields = REQUIRED_PARAMETER_FIELDS - parameter.keys()
        for field in sorted(missing_fields):
            add_message(errors, location, f"missing required field {field!r}")
        if missing_fields:
            continue

        parameter_id = parameter["id"]
        if not isinstance(parameter_id, str) or not parameter_id.strip():
            add_message(errors, location, "id must be a non-empty string")
            continue
        if parameter_id in parameter_ids:
            add_message(
                errors,
                location,
                f"duplicate parameter id {parameter_id!r} "
                f"(first at parameters[{parameter_ids[parameter_id]}])",
            )
        else:
            parameter_ids[parameter_id] = index

        for field in ("parameter_name", "full_name", "description"):
            value = parameter[field]
            if not isinstance(value, str) or not value.strip():
                add_message(errors, location, f"{field} must be a non-empty string")

        parameter_name = parameter["parameter_name"]
        if isinstance(parameter_name, str) and parameter_name.strip():
            normalized_name = normalize_label(parameter_name)
            if normalized_name in parameter_names:
                add_message(
                    warnings,
                    location,
                    f"duplicate normalized parameter name {parameter_name!r}",
                )
            else:
                parameter_names[normalized_name] = index

        aliases = parameter["aliases"]
        if not isinstance(aliases, list) or not aliases:
            add_message(errors, location, "aliases must be a non-empty list")
            aliases = []
        local_labels: set[str] = set()
        declared_aliases: set[str] = set()
        for alias in aliases:
            if not isinstance(alias, str) or not alias.strip():
                continue
            normalized_alias = normalize_label(alias)
            if normalized_alias in declared_aliases:
                add_message(
                    warnings,
                    location,
                    f"duplicate normalized alias within aliases list: {alias!r}",
                )
            declared_aliases.add(normalized_alias)

        labels = [parameter_name, *aliases] if isinstance(parameter_name, str) else aliases
        for alias in aliases:
            if not isinstance(alias, str) or not alias.strip():
                add_message(errors, location, "aliases must contain non-empty strings")
                continue
        for label in labels:
            if not isinstance(label, str) or not label.strip():
                continue
            normalized_alias = normalize_label(label)
            if not normalized_alias:
                add_message(errors, location, f"alias {label!r} has no searchable characters")
                continue
            if normalized_alias in local_labels:
                continue
            local_labels.add(normalized_alias)
            existing = alias_owners.get(normalized_alias)
            if existing and existing[0] != parameter_id:
                add_message(
                    errors,
                    location,
                    f"conflicting alias {label!r}; also maps to "
                    f"{existing[0]!r} at parameters[{existing[1]}] as {existing[2]!r}",
                )
            else:
                alias_owners[normalized_alias] = (parameter_id, index, label)

        unit = parameter["unit"]
        if not isinstance(unit, str) or not unit.strip():
            add_message(errors, location, "unit must be a non-empty string")
        elif unit != unit.strip():
            add_message(warnings, location, f"unit has surrounding whitespace: {unit!r}")
        elif not UNIT_PATTERN.fullmatch(unit):
            add_message(errors, location, f"unit contains unsupported characters: {unit!r}")
        elif re.search(r"\s{2,}", unit):
            add_message(warnings, location, f"unit has repeated internal whitespace: {unit!r}")
        if isinstance(unit, str) and unit.strip():
            counts["units"] += 1

        category_id = parameter["category_id"]
        if not isinstance(category_id, str) or category_id not in category_ids:
            add_message(errors, location, f"unknown category id {category_id!r}")

        ranges = parameter["typical_reference_ranges"]
        if not isinstance(ranges, list) or not ranges:
            add_message(errors, location, "typical_reference_ranges must be a non-empty list")
            continue

        for range_index, reference_range in enumerate(ranges):
            range_location = f"{location}.typical_reference_ranges[{range_index}]"
            if not isinstance(reference_range, dict):
                add_message(errors, range_location, "expected an object")
                continue
            missing_range_fields = REQUIRED_RANGE_FIELDS - reference_range.keys()
            for field in sorted(missing_range_fields):
                add_message(errors, range_location, f"missing required field {field!r}")
            if missing_range_fields:
                continue

            for field in ("population", "range_type", "qualifications"):
                value = reference_range[field]
                if not isinstance(value, str) or not value.strip():
                    add_message(errors, range_location, f"{field} must be a non-empty string")
            range_type = reference_range["range_type"]
            if isinstance(range_type, str) and range_type not in KNOWN_RANGE_TYPES:
                add_message(
                    warnings,
                    range_location,
                    f"unrecognized range_type {range_type!r}",
                )
            if isinstance(reference_range["qualifications"], str) and reference_range[
                "qualifications"
            ].strip():
                counts["qualified_ranges"] += 1

            range_value = reference_range["range"]
            if range_value is None:
                counts["null_ranges"] += 1
                if reference_range["range_type"] not in NULL_RANGE_TYPES:
                    add_message(
                        warnings,
                        range_location,
                        "null range should be marked assay-specific, risk-based, "
                        "or without a single reference interval",
                    )
                continue
            if not isinstance(range_value, str) or not range_value.strip():
                add_message(errors, range_location, "range must be a non-empty string or null")
                continue

            parsed_range = parse_range(range_value)
            if parsed_range is None:
                add_message(
                    errors,
                    range_location,
                    f"unrecognized or invalid reference range {range_value!r}",
                )
            else:
                counts["ranges"] += 1
                counts[f"{parsed_range[0]}_ranges"] += 1

    sources = data.get("sources")
    if not isinstance(sources, list):
        add_message(errors, "sources", "expected a list")
        sources = []
    counts["sources"] = len(sources)

    source_ids: set[str] = set()
    for index, source in enumerate(sources):
        location = f"sources[{index}]"
        if not isinstance(source, dict):
            add_message(errors, location, "expected an object")
            continue
        for field in sorted(REQUIRED_SOURCE_FIELDS - source.keys()):
            add_message(errors, location, f"missing required field {field!r}")
        if REQUIRED_SOURCE_FIELDS - source.keys():
            continue
        source_id = source["id"]
        if not isinstance(source_id, str) or not source_id.strip():
            add_message(errors, location, "id must be a non-empty string")
        elif source_id in source_ids:
            add_message(errors, location, f"duplicate source id {source_id!r}")
        else:
            source_ids.add(source_id)

        for field in ("title", "organization", "used_for"):
            value = source[field]
            if not isinstance(value, str) or not value.strip():
                add_message(errors, location, f"{field} must be a non-empty string")

        url = source["url"]
        if not isinstance(url, str):
            add_message(errors, location, "url must be a string")
        else:
            parsed_url = urlparse(url)
            if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
                add_message(errors, location, f"invalid source URL {url!r}")

    return errors, warnings, counts


def main() -> int:
    """Load the catalog, print a complete report, and return a process status."""
    print("MediExplain AI blood-test database validation")
    print(f"File: {DATABASE_PATH}")
    try:
        with DATABASE_PATH.open(encoding="utf-8") as database_file:
            data = json.load(database_file)
    except FileNotFoundError:
        print("ERROR: database file was not found.")
        return 1
    except json.JSONDecodeError as error:
        print(
            f"ERROR: invalid JSON at line {error.lineno}, "
            f"column {error.colno}: {error.msg}"
        )
        return 1

    errors, warnings, counts = validate_database(data)
    if counts["parameters"] != EXPECTED_PARAMETER_COUNT:
        warnings.append(
            f"parameter count: expected {EXPECTED_PARAMETER_COUNT} per the audit "
            f"request, found {counts['parameters']}"
        )

    print(
        "Summary: "
        f"{counts['parameters']} parameters, {counts['categories']} categories, "
        f"{counts['sources']} sources, {counts['units']} units, "
        f"{counts['ranges']} parsed ranges "
        f"({counts['bounded_ranges']} bounded, {counts['threshold_ranges']} "
        f"threshold, {counts['single_value_ranges']} single-value), "
        f"{counts['null_ranges']} null ranges, "
        f"{counts['qualified_ranges']} qualified ranges"
    )
    print(f"Errors: {len(errors)}")
    for error in errors:
        print(f"  ERROR: {error}")
    print(f"Warnings: {len(warnings)}")
    for warning in warnings:
        print(f"  WARNING: {warning}")

    if errors:
        print("RESULT: FAILED")
        return 1
    if warnings:
        print("RESULT: PASSED WITH WARNINGS")
    else:
        print("RESULT: PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
