"""Load and query the human blood-test parameter catalog."""

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import re
from typing import Any, Literal, Mapping
import unicodedata


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DATABASE_PATH = _PROJECT_ROOT / "data" / "blood_test_parameters.json"

_REQUIRED_ROOT_FIELDS = {
    "schema_version",
    "dataset_name",
    "scope",
    "safety_and_range_policy",
    "categories",
    "parameters",
    "sources",
}
_REQUIRED_PARAMETER_FIELDS = {
    "id",
    "parameter_name",
    "full_name",
    "aliases",
    "unit",
    "category_id",
    "typical_reference_ranges",
    "description",
}
_REQUIRED_RANGE_FIELDS = {
    "population",
    "range",
    "range_type",
    "qualifications",
}
_REQUIRED_CATEGORY_FIELDS = {"id", "name", "description"}
_REQUIRED_SOURCE_FIELDS = {"id", "title", "organization", "url", "used_for"}
_ALLOWED_RANGE_TYPES = frozenset(
    {
        "assay_specific",
        "clinical_decision_threshold",
        "general_reference_threshold",
        "no_single_reference_interval",
        "nutrient_adequacy_reference",
        "reference_interval",
        "risk_based_target",
    }
)
@dataclass(frozen=True)
class ParameterResolution:
    """The result of resolving a name, including ambiguity or unit mismatch."""

    status: Literal["matched", "ambiguous", "not_found", "unit_mismatch"]
    parameter: dict[str, Any] | None
    candidates: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class _CatalogIndexes:
    by_id: dict[str, dict[str, Any]]
    by_name: dict[str, tuple[dict[str, Any], ...]]


def _normalize_parameter_name(name: str) -> str:
    """Ignore case and harmless punctuation while preserving meaningful suffixes."""
    normalized = unicodedata.normalize("NFKC", name).casefold().strip()
    tokens = re.findall(r"[a-z0-9]+|%", normalized)
    return "".join(tokens)


def _normalize_unit_for_match(unit: str) -> str:
    """Normalize outer and slash-adjacent spaces without changing unit symbols."""
    return re.sub(r"\s*/\s*", "/", unit.strip())


def units_are_equivalent(first: str, second: str, parameter_id: str) -> bool:
    """Compare exact spellings plus the approved RBC same-scale unit alias."""
    first_normalized = _normalize_unit_for_match(first)
    second_normalized = _normalize_unit_for_match(second)
    if first_normalized == second_normalized:
        return True
    return (
        parameter_id == "rbc_count"
        and {first_normalized, second_normalized}
        == {"million/uL", "10^6/uL"}
    )


def _require_nonempty_string(record: Mapping[str, Any], field: str, location: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location}.{field} must be a non-empty string.")
    return value


def _validate_catalog(data: Any) -> None:
    """Fail clearly when the catalog does not have the expected usable shape."""
    if not isinstance(data, dict):
        raise ValueError("The parameter database root must be a JSON object.")

    missing_root_fields = _REQUIRED_ROOT_FIELDS - data.keys()
    if missing_root_fields:
        missing = ", ".join(sorted(missing_root_fields))
        raise ValueError(f"The parameter database is missing root fields: {missing}.")

    for field in ("schema_version", "dataset_name", "scope"):
        _require_nonempty_string(data, field, "root")
    if not isinstance(data["safety_and_range_policy"], dict):
        raise ValueError("safety_and_range_policy must be a JSON object.")
    for field in ("categories", "parameters", "sources"):
        if not isinstance(data[field], list):
            raise ValueError(f"{field} must be a JSON list.")
        if not data[field]:
            raise ValueError(f"{field} must not be empty.")

    category_ids: set[str] = set()
    for index, category in enumerate(data["categories"]):
        location = f"categories[{index}]"
        if not isinstance(category, dict):
            raise ValueError(f"{location} must be a JSON object.")
        missing = _REQUIRED_CATEGORY_FIELDS - category.keys()
        if missing:
            raise ValueError(
                f"{location} is missing fields: {', '.join(sorted(missing))}."
            )
        category_id = _require_nonempty_string(category, "id", location)
        _require_nonempty_string(category, "name", location)
        _require_nonempty_string(category, "description", location)
        if category_id in category_ids:
            raise ValueError(f"Duplicate category id {category_id!r}.")
        category_ids.add(category_id)

    parameter_ids: set[str] = set()
    for index, parameter in enumerate(data["parameters"]):
        location = f"parameters[{index}]"
        if not isinstance(parameter, dict):
            raise ValueError(f"{location} must be a JSON object.")
        missing = _REQUIRED_PARAMETER_FIELDS - parameter.keys()
        if missing:
            raise ValueError(
                f"{location} is missing fields: {', '.join(sorted(missing))}."
            )

        parameter_id = _require_nonempty_string(parameter, "id", location)
        for field in ("parameter_name", "full_name", "unit", "description"):
            _require_nonempty_string(parameter, field, location)
        if parameter_id in parameter_ids:
            raise ValueError(f"Duplicate parameter id {parameter_id!r}.")
        parameter_ids.add(parameter_id)

        category_id = parameter["category_id"]
        if not isinstance(category_id, str) or not category_id.strip():
            raise ValueError(f"{location}.category_id must be a non-empty string.")
        if category_id not in category_ids:
            raise ValueError(
                f"{location}.category_id refers to unknown category "
                f"{category_id!r}."
            )
        aliases = parameter["aliases"]
        if not isinstance(aliases, list) or not aliases:
            raise ValueError(f"{location}.aliases must be a non-empty JSON list.")
        for alias_index, alias in enumerate(aliases):
            if not isinstance(alias, str) or not alias.strip():
                raise ValueError(
                    f"{location}.aliases[{alias_index}] must be a non-empty string."
                )
            if not _normalize_parameter_name(alias):
                raise ValueError(
                    f"{location}.aliases[{alias_index}] has no searchable characters."
                )

        ranges = parameter["typical_reference_ranges"]
        if not isinstance(ranges, list) or not ranges:
            raise ValueError(
                f"{location}.typical_reference_ranges must be a non-empty JSON list."
            )
        for range_index, reference_range in enumerate(ranges):
            range_location = f"{location}.typical_reference_ranges[{range_index}]"
            if not isinstance(reference_range, dict):
                raise ValueError(f"{range_location} must be a JSON object.")
            missing_range_fields = _REQUIRED_RANGE_FIELDS - reference_range.keys()
            if missing_range_fields:
                raise ValueError(
                    f"{range_location} is missing fields: "
                    f"{', '.join(sorted(missing_range_fields))}."
                )
            for field in ("population", "range_type", "qualifications"):
                _require_nonempty_string(reference_range, field, range_location)
            range_type = reference_range["range_type"]
            if range_type not in _ALLOWED_RANGE_TYPES:
                allowed_values = ", ".join(sorted(_ALLOWED_RANGE_TYPES))
                raise ValueError(
                    f"{range_location}.range_type has unsupported value "
                    f"{range_type!r}. Expected one of: {allowed_values}."
                )
            range_value = reference_range["range"]
            if range_value is not None and (
                not isinstance(range_value, str) or not range_value.strip()
            ):
                raise ValueError(
                    f"{range_location}.range must be a non-empty string or null."
                )

    source_ids: set[str] = set()
    for index, source in enumerate(data["sources"]):
        location = f"sources[{index}]"
        if not isinstance(source, dict):
            raise ValueError(f"{location} must be a JSON object.")
        missing = _REQUIRED_SOURCE_FIELDS - source.keys()
        if missing:
            raise ValueError(
                f"{location} is missing fields: {', '.join(sorted(missing))}."
            )
        source_id = _require_nonempty_string(source, "id", location)
        for field in ("title", "organization", "url", "used_for"):
            _require_nonempty_string(source, field, location)
        if source_id in source_ids:
            raise ValueError(f"Duplicate source id {source_id!r}.")
        source_ids.add(source_id)


@lru_cache(maxsize=1)
def _load_cached_catalog() -> dict[str, Any]:
    """Read and validate the catalog once, using this module's project path."""
    try:
        with _DATABASE_PATH.open(encoding="utf-8") as database_file:
            data = json.load(database_file)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Blood-test parameter database was not found at {_DATABASE_PATH}."
        ) from error
    except json.JSONDecodeError as error:
        raise ValueError(
            "Blood-test parameter database contains invalid JSON at "
            f"line {error.lineno}, column {error.colno}: {error.msg}."
        ) from error

    _validate_catalog(data)
    return data


def _build_indexes(data: Mapping[str, Any]) -> _CatalogIndexes:
    """Build exact name indexes while retaining all matching candidates."""
    by_id: dict[str, dict[str, Any]] = {}
    mutable_by_name: dict[str, dict[str, dict[str, Any]]] = {}

    for parameter in data["parameters"]:
        parameter_id = parameter["id"]
        by_id[parameter_id] = parameter
        names = (parameter["parameter_name"], *parameter["aliases"])
        for name in names:
            normalized_name = _normalize_parameter_name(name)
            if not normalized_name:
                continue
            mutable_by_name.setdefault(normalized_name, {})[parameter_id] = parameter

    by_name = {
        name: tuple(candidates.values())
        for name, candidates in mutable_by_name.items()
    }
    return _CatalogIndexes(by_id=by_id, by_name=by_name)


@lru_cache(maxsize=1)
def _get_catalog_indexes() -> _CatalogIndexes:
    return _build_indexes(_load_cached_catalog())


def load_parameter_catalog() -> dict[str, Any]:
    """Return a copy of the validated catalog without altering its metadata."""
    return deepcopy(_load_cached_catalog())


def clear_parameter_catalog_cache() -> None:
    """Clear cached file contents and indexes, primarily for tests."""
    _get_catalog_indexes.cache_clear()
    _load_cached_catalog.cache_clear()


def get_parameter_by_id(parameter_id: str) -> dict[str, Any] | None:
    """Return a catalog record by exact ID, or None when the ID is unknown."""
    if not isinstance(parameter_id, str) or not parameter_id.strip():
        return None
    parameter = _get_catalog_indexes().by_id.get(parameter_id.strip())
    return deepcopy(parameter) if parameter is not None else None


def find_parameter_candidates(name: str) -> list[dict[str, Any]]:
    """Return every exact normalized name/alias match, without guessing."""
    if not isinstance(name, str) or not name.strip():
        return []
    normalized_name = _normalize_parameter_name(name)
    if not normalized_name:
        return []
    candidates = _get_catalog_indexes().by_name.get(normalized_name, ())
    return [deepcopy(parameter) for parameter in candidates]


def _resolve_candidates(
    candidates: tuple[dict[str, Any], ...],
    unit: str | None,
) -> ParameterResolution:
    if not candidates:
        return ParameterResolution("not_found", None, ())

    normalized_unit = (
        _normalize_unit_for_match(unit)
        if isinstance(unit, str)
        else None
    )
    has_unit = bool(normalized_unit)
    invalid_unit_type = unit is not None and normalized_unit is None
    if not has_unit and not invalid_unit_type and len(candidates) == 1:
        parameter = deepcopy(candidates[0])
        return ParameterResolution("matched", parameter, (deepcopy(parameter),))

    if invalid_unit_type:
        return ParameterResolution(
            "unit_mismatch",
            None,
            tuple(deepcopy(parameter) for parameter in candidates),
        )

    if normalized_unit:
        unit_matches = tuple(
            parameter
            for parameter in candidates
            if units_are_equivalent(
                parameter["unit"],
                normalized_unit,
                parameter["id"],
            )
        )
        if len(unit_matches) == 1:
            parameter = deepcopy(unit_matches[0])
            return ParameterResolution("matched", parameter, (deepcopy(parameter),))
        if not unit_matches:
            return ParameterResolution(
                "unit_mismatch",
                None,
                tuple(deepcopy(parameter) for parameter in candidates),
            )
        candidates = unit_matches

    return ParameterResolution(
        "ambiguous",
        None,
        tuple(deepcopy(parameter) for parameter in candidates),
    )


def resolve_parameter(
    name: str,
    unit: str | None = None,
) -> ParameterResolution:
    """Resolve a name and explicitly report supplied units that do not match."""
    candidates = find_parameter_candidates(name)
    return _resolve_candidates(tuple(candidates), unit)


def get_applicable_ranges(
    parameter_id: str,
    context: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return ranges selected by an exact population label, without inference.

    With no context, only an explicitly all-ages range is returned. Otherwise,
    context must contain a ``population`` value exactly matching the database's
    population label, ignoring case and surrounding/repeated whitespace.
    Age and free-text qualifications are never interpreted. Explicit sex and
    pregnancy values only reject conflicts with information in the population
    label; they do not select a range or infer applicability.
    """
    parameter = get_parameter_by_id(parameter_id)
    if parameter is None:
        return []

    if context is None:
        population = "all ages"
        context_values: Mapping[str, Any] = {}
    elif isinstance(context, Mapping):
        population_value = context.get("population")
        if not isinstance(population_value, str) or not population_value.strip():
            return []
        population = re.sub(r"\s+", " ", population_value.strip()).casefold()
        context_values = context
    else:
        return []

    applicable = []
    for reference_range in parameter["typical_reference_ranges"]:
        range_population = re.sub(
            r"\s+",
            " ",
            reference_range["population"].strip(),
        ).casefold()
        if range_population != population:
            continue

        population_label = reference_range["population"].casefold()
        population_sexes: set[str] = set()
        if re.search(r"\bmale(s)?\b", population_label):
            population_sexes.add("male")
        if re.search(r"\bfemale(s)?\b", population_label):
            population_sexes.add("female")

        supplied_sex = context_values.get("sex")
        if supplied_sex is not None:
            if not isinstance(supplied_sex, str):
                continue
            normalized_sex = supplied_sex.strip().casefold()
            if normalized_sex in {"m", "male", "man"}:
                normalized_sex = "male"
            elif normalized_sex in {"f", "female", "woman"}:
                normalized_sex = "female"
            if population_sexes and normalized_sex not in population_sexes:
                continue

        supplied_pregnancy = context_values.get("pregnancy")
        if supplied_pregnancy is not None:
            if isinstance(supplied_pregnancy, bool):
                is_pregnant = supplied_pregnancy
            elif isinstance(supplied_pregnancy, str):
                normalized_pregnancy = re.sub(
                    r"[\s_-]+", "", supplied_pregnancy
                ).casefold()
                if normalized_pregnancy in {"pregnant", "yes", "true"}:
                    is_pregnant = True
                elif normalized_pregnancy in {
                    "notpregnant",
                    "nonpregnant",
                    "no",
                    "false",
                }:
                    is_pregnant = False
                else:
                    continue
            else:
                continue

            population_is_nonpregnant = bool(
                re.search(r"\bnon[\s-]?pregnant\b", population_label)
            )
            population_is_pregnant = (
                "pregnant" in population_label and not population_is_nonpregnant
            )
            if (
                is_pregnant and population_is_nonpregnant
            ) or (
                not is_pregnant and population_is_pregnant
            ):
                continue

        applicable.append(deepcopy(reference_range))
    return applicable
