"""Compatibility adapter for the JSON-backed human blood-test catalog."""

from dataclasses import dataclass
import re
import unicodedata

from modules.parameter_database import (
    ParameterResolution,
    find_parameter_candidates,
    get_parameter_by_id,
    load_parameter_catalog,
    resolve_parameter,
    units_are_equivalent,
)


@dataclass(frozen=True)
class TestDefinition:
    """Compatibility view of one catalog parameter."""

    canonical_name: str
    panel: str
    aliases: tuple[str, ...]
    units: tuple[str, ...]
    parameter_id: str = ""
    category_id: str = ""


_PANEL_BY_CATEGORY_ID = {
    "cbc": "CBC",
    "lft": "LFT",
    "kft": "KFT",
    "lipid": "Lipid profile",
    "thyroid": "Thyroid profile",
    "diabetes": "Glucose",
    "iron": "Iron profile",
    "vitamin": "Vitamin profile",
}

# Keep the existing sample-report label without duplicating parameter metadata.
_LEGACY_NAME_ALIASES = {
    "Blood Glucose (Fasting)": "fasting_glucose",
}


def normalize_test_name(test_name: str) -> str:
    """Return a punctuation-tolerant key while preserving meaningful tokens."""
    normalized = unicodedata.normalize("NFKC", test_name).casefold().strip()
    return "".join(re.findall(r"[a-z0-9]+|%", normalized))


def _panel_for_parameter(parameter: dict) -> str:
    if parameter["id"] == "hba1c":
        return "HbA1c"
    return _PANEL_BY_CATEGORY_ID.get(
        parameter["category_id"],
        parameter["category_id"],
    )


def _definition_from_parameter(parameter: dict) -> TestDefinition:
    return TestDefinition(
        canonical_name=parameter["parameter_name"],
        panel=_panel_for_parameter(parameter),
        aliases=tuple(parameter["aliases"]),
        units=(parameter["unit"],),
        parameter_id=parameter["id"],
        category_id=parameter["category_id"],
    )


_catalog = load_parameter_catalog()
_definitions_by_id = {
    parameter["id"]: _definition_from_parameter(parameter)
    for parameter in _catalog["parameters"]
}
TEST_DEFINITIONS = tuple(_definitions_by_id.values())


def get_test_definition_by_id(parameter_id: str) -> TestDefinition | None:
    """Return the compatibility definition for a stable catalog parameter ID."""
    return _definitions_by_id.get(parameter_id)


def get_test_parameter_by_id(parameter_id: str) -> dict | None:
    """Return a copy of the complete JSON-backed parameter record."""
    return get_parameter_by_id(parameter_id)


def _resolve_legacy_alias(
    test_name: str,
    unit: str | None,
) -> ParameterResolution | None:
    parameter_id = next(
        (
            parameter_id
            for alias, parameter_id in _LEGACY_NAME_ALIASES.items()
            if normalize_test_name(alias) == normalize_test_name(test_name)
        ),
        None,
    )
    if parameter_id is None:
        return None

    parameter = get_parameter_by_id(parameter_id)
    if parameter is None:
        return None
    candidates = (parameter,)
    if isinstance(unit, str) and unit.strip():
        if not units_are_equivalent(
            unit,
            parameter["unit"],
            parameter["id"],
        ):
            return ParameterResolution("unit_mismatch", None, candidates)
    elif unit is not None and not isinstance(unit, str):
        return ParameterResolution("unit_mismatch", None, candidates)
    return ParameterResolution("matched", parameter, candidates)


def resolve_test_definition(
    test_name: str,
    unit: str | None = None,
) -> ParameterResolution:
    """Resolve an exact catalog name or alias, optionally using its exact unit."""
    resolution = resolve_parameter(test_name, unit)
    if resolution.status != "not_found":
        return resolution
    legacy_resolution = _resolve_legacy_alias(test_name, unit)
    return legacy_resolution or resolution


def find_test_candidates(test_name: str) -> tuple[TestDefinition, ...]:
    """Return all exact name/alias candidates without selecting ambiguities."""
    candidates = find_parameter_candidates(test_name)
    if not candidates:
        legacy_resolution = _resolve_legacy_alias(test_name, None)
        candidates = (
            [legacy_resolution.parameter]
            if legacy_resolution is not None and legacy_resolution.parameter
            else []
        )
    return tuple(
        _definitions_by_id[parameter["id"]]
        for parameter in candidates
        if parameter["id"] in _definitions_by_id
    )


def find_test_definition(test_name: str) -> TestDefinition | None:
    """Return a unique exact name/alias match, or None if absent or ambiguous."""
    resolution = resolve_test_definition(test_name)
    if resolution.status != "matched" or resolution.parameter is None:
        return None
    return _definitions_by_id.get(resolution.parameter["id"])


def _alias_to_pattern(alias: str) -> str:
    """Allow flexible whitespace while keeping the full alias token sequence."""
    return r"\s+".join(re.escape(part) for part in alias.split())


_pattern_aliases = {
    name
    for definition in TEST_DEFINITIONS
    for name in (definition.canonical_name, *definition.aliases)
}
_pattern_aliases.update(_LEGACY_NAME_ALIASES)
_alternatives = "|".join(
    _alias_to_pattern(alias)
    for alias in sorted(_pattern_aliases, key=len, reverse=True)
)
LAB_TEST_NAME_PATTERN = re.compile(
    rf"(?<!\w)(?P<name>{_alternatives})(?![\w/])",
    re.IGNORECASE,
)
