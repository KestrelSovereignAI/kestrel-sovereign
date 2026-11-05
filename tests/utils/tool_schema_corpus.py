"""JSON Schemas an isolated service may advertise for structured input (#3925).

Each is a complete ``ToolMetadata.input_schema``. Together they cover the
shapes the flat ``ToolParameter`` projection lost before the model saw them:
nested required objects, arrays of objects, nullable types, enums, ``anyOf``,
closed objects and a local ``$ref``.
"""

#: The issue's reproduction: ``record.id`` is required, both objects are
#: closed, and ``note`` is nullable.
ISSUE_3925_SCHEMA = {
    "type": "object",
    "properties": {
        "record": {
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
            "additionalProperties": False,
        },
        "note": {"type": ["string", "null"]},
    },
    "required": ["record"],
    "additionalProperties": False,
}

SCHEMA_CORPUS = {
    "nested required object": {
        "type": "object",
        "properties": {
            "record": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Record id"},
                    "label": {"type": "string"},
                },
                "required": ["id"],
            },
        },
        "required": ["record"],
    },
    "array of objects": {
        "type": "object",
        "properties": {
            "lines": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "sku": {"type": "string"},
                        "qty": {"type": "integer", "minimum": 1},
                    },
                    "required": ["sku", "qty"],
                },
            },
        },
        "required": ["lines"],
    },
    "nullable type": {
        "type": "object",
        "properties": {
            "note": {"type": ["string", "null"], "description": "Optional note"},
        },
    },
    "enum": {
        "type": "object",
        "properties": {"mode": {"type": "string", "enum": ["fast", "safe"]}},
        "required": ["mode"],
    },
    "anyOf": {
        "type": "object",
        "properties": {
            "target": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
        },
        "required": ["target"],
    },
    "closed objects": {
        "type": "object",
        "properties": {
            "record": {
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
                "additionalProperties": False,
            },
        },
        "required": ["record"],
        "additionalProperties": False,
    },
    "local $ref": {
        "type": "object",
        "$defs": {
            "Address": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
        "properties": {
            "home": {"$ref": "#/$defs/Address"},
            "work": {"$ref": "#/$defs/Address", "description": "Work address"},
        },
        "required": ["home"],
    },
    "issue #3925 reproduction": ISSUE_3925_SCHEMA,
}
