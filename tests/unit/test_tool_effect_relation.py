"""Closed PostgreSQL catalog grammar; native behavior is covered by backend tests."""

from copy import deepcopy

import pytest

from kestrel_sovereign.storage.db.tool_effect_relation import (
    _check_program, _verify_tool_effect_programs,
)
from kestrel_sovereign.tool_effect_store import (
    TOOL_EFFECT_OPERATIONS_COLUMNS, TOOL_EFFECT_CLAIMS_COLUMNS,
)


def catalog():
    tables = [dict(oid=oid, name=name, columns=list(columns),
        numbers=list(range(1, len(columns) + 1)),
        required=[column != 'parent_claim_id' for column in columns])
        for oid, name, columns in [(101, 'tool_effect_operations', TOOL_EFFECT_OPERATIONS_COLUMNS),
            (102, 'tool_effect_claims', TOOL_EFFECT_CLAIMS_COLUMNS)]]
    def constraint(oid, owner, kind, **fields):
        return dict(oid=oid, owner=owner, kind=kind, validated=True, enforced=True, local=True,
            inherited=0, parent=0, deferred=False, deferrable=False, expression=None,
            builtin_expression=True, builtin_equality=True, **fields)
    constraints = [
        constraint(201, 101, 'p', keys=[1, 2], index=301),
        constraint(202, 102, 'p', keys=[1, 2, 3], index=302),
        constraint(203, 102, 'f', keys=[1, 2], foreign_keys=[1, 2], referenced=101,
            update_action='a', delete_action='a', match='s', index=301),
    ]
    for oid, owner, expression in [
        (204, 101, "(state = ANY (ARRAY['OPEN'::text, 'CLOSED'::text]))"),
        (205, 101, '(latched = ANY (ARRAY[(0)::bigint, (1)::bigint]))'),
        (206, 102, "(state = ANY (ARRAY['STARTED'::text, 'RETURNED_OK'::text, 'RETURNED_ERROR'::text, 'UNCONFIRMED'::text, 'NOT_EXECUTED'::text]))"),
    ]:
        row = constraint(oid, owner, 'c')
        row['expression'] = expression
        constraints.append(row)
    triggers = [dict(owner=owner, referenced=other, function=function, events=events,
        internal=True, builtin_function=True, constraint=203, index=301, enabled='O',
        deferrable=False, deferred=False, parent=0, nargs=0, attributes=[], args=b'',
        when=None, old_table=None, new_table=None)
        for owner, other, function, events in [
            (101, 102, 'RI_FKey_noaction_del', 9), (101, 102, 'RI_FKey_noaction_upd', 17),
            (102, 101, 'RI_FKey_check_ins', 5), (102, 101, 'RI_FKey_check_upd', 17)]]
    return tables, constraints, triggers


def test_exact_canonical_checks_and_four_builtin_fk_triggers_are_admitted():
    assert _verify_tool_effect_programs(*catalog())


@pytest.mark.parametrize('collection,index,key,value', [
    (0, 0, 'name', 'foreign_table'), (0, 1, 'oid', 999),
    (0, 1, 'required', [True] * 8),
    (1, 0, 'keys', [2, 1]), (1, 0, 'deferrable', True),
    (1, 0, 'enforced', False),
    (1, 2, 'referenced', 999), (1, 2, 'keys', [1, 3]),
    (1, 2, 'builtin_equality', False), (1, 2, 'update_action', 'c'),
    (1, 2, 'index', 999), (1, 3, 'builtin_expression', False),
    (1, 3, 'expression', "evil(state) = ANY (ARRAY['OPEN','CLOSED'])"),
    (1, 3, 'expression', "state OPERATOR(evil.=) ANY (ARRAY['OPEN','CLOSED'])"),
    (1, 4, 'expression', '(latched IN (0,1) OR evil())'),
    (2, 0, 'function', 'application_trigger'), (2, 0, 'internal', False),
    (2, 0, 'builtin_function', False), (2, 0, 'constraint', 999),
    (2, 0, 'enabled', 'D'), (2, 0, 'args', b'callback'),
    (2, 0, 'when', 'evil()'), (2, 0, 'attributes', [1]),
    (2, 0, 'events', 17), (2, 0, 'deferrable', True),
])
def test_any_foreign_program_or_constraint_pair_is_refused(collection, index, key, value):
    rows = deepcopy(catalog())
    rows[collection][index][key] = value
    assert not _verify_tool_effect_programs(*rows)


@pytest.mark.parametrize('collection', [0, 1, 2])
def test_missing_or_duplicated_catalog_members_are_refused(collection):
    for duplicate in (False, True):
        rows = deepcopy(catalog())
        rows[collection][:] = rows[collection][:-1] + ([rows[collection][0]] if duplicate else [])
        assert not _verify_tool_effect_programs(*rows)


def test_check_normalization_keeps_foreign_program_names():
    assert _check_program("(state = ANY (ARRAY['OPEN'::pg_catalog.text, 'CLOSED'::text]))") == "state=ANYARRAY['OPEN','CLOSED']"
    assert 'evil' in _check_program('evil(state) OPERATOR(evil.=) ANY (ARRAY[0,1])')
