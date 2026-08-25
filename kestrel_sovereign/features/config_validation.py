"""Transport-neutral validation of feature configuration against its schema.

One rule, two doors. Feature configuration arrives from the HTTP configuration
endpoint and, since #3008, from the host's declarative
``[features.<package>.config]`` block at boot. Both must apply the same rule,
and a rule that raises ``HTTPException`` can only be used by one of them — so
the check lives here and raises a plain error, and the HTTP layer converts it
at its own seam.

Duplicating the check in the boot path instead would have been the obvious
move and the wrong one: two copies of a validation rule drift, and the copy
that drifts is the one nobody is looking at.
"""

from __future__ import annotations

from typing import Any, Dict


class FeatureConfigInvalid(ValueError):
    """A configuration value does not satisfy the feature's declared schema."""


_TYPE_MAP: Dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


def validate_feature_config(config: Dict[str, Any], schema: Dict[str, Any]) -> None:
    """Check required fields, types, numeric bounds and enums.

    Raises:
        FeatureConfigInvalid: with a message naming the offending field.
    """
    required = schema.get("required", [])
    properties = schema.get("properties", {})

    for field_name in required:
        if field_name not in config:
            raise FeatureConfigInvalid(
                f"Missing required config field: '{field_name}'"
            )

    for key, value in config.items():
        if key not in properties:
            continue
        prop_schema = properties[key]

        expected_type_name = prop_schema.get("type")
        if expected_type_name and expected_type_name in _TYPE_MAP:
            expected = _TYPE_MAP[expected_type_name]
            # bool is an int subclass; an integer field must not accept True.
            if expected_type_name in ("integer", "number") and isinstance(
                value, bool
            ):
                raise FeatureConfigInvalid(
                    f"Config field '{key}' must be {expected_type_name}, got bool"
                )
            if not isinstance(value, expected):
                raise FeatureConfigInvalid(
                    f"Config field '{key}' must be {expected_type_name}, "
                    f"got {type(value).__name__}"
                )

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            minimum = prop_schema.get("minimum")
            if minimum is not None and value < minimum:
                raise FeatureConfigInvalid(
                    f"Config field '{key}' must be >= {minimum}, got {value}"
                )
            maximum = prop_schema.get("maximum")
            if maximum is not None and value > maximum:
                raise FeatureConfigInvalid(
                    f"Config field '{key}' must be <= {maximum}, got {value}"
                )

        enum_values = prop_schema.get("enum")
        if enum_values is not None and value not in enum_values:
            raise FeatureConfigInvalid(
                f"Config field '{key}' must be one of {enum_values}, got {value!r}"
            )
