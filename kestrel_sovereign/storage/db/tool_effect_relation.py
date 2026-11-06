"""Closed catalog proof for the native tool-effect journal's constraint pair.

This observes metadata only. The canonical native row writer owns locking,
program seals, mutation, COMMIT, rollback and physical connection retirement.
"""

import re


def _check_program(expression):
    # pg_get_expr qualifies a non-visible operator/function. Only these exact
    # enum expressions and their builtin scalar casts are admitted. Catalog
    # dependencies below separately reject non-builtin expression programs.
    return re.sub(r'\s+|[()]|::(?:pg_catalog\.)?(?:text|bigint)\b', '', expression)


def _verify_tool_effect_programs(tables, constraints, triggers):
    from kestrel_sovereign.tool_effect_store import (
        TOOL_EFFECT_OPERATIONS_COLUMNS, TOOL_EFFECT_CLAIMS_COLUMNS,
    )

    profiles = {'tool_effect_operations': TOOL_EFFECT_OPERATIONS_COLUMNS,
                'tool_effect_claims': TOOL_EFFECT_CLAIMS_COLUMNS}
    if len(tables) != 2 or {row['name'] for row in tables} != set(profiles):
        return False
    by_name = {row['name']: row for row in tables}
    by_oid = {row['oid']: row for row in tables}
    for name, row in by_name.items():
        if (set(row['columns']) != set(profiles[name])
                or len(row['numbers']) != len(profiles[name])
                or len(row['required']) != len(profiles[name])):
            return False
        if any(required != (column != 'parent_claim_id')
               for column, required in zip(row['columns'], row['required'])):
            return False
    keys = {row['oid']: dict(zip(row['columns'], row['numbers'])) for row in tables}
    expected_checks = {
        'tool_effect_operations': {"state=ANYARRAY['OPEN','CLOSED']", 'latched=ANYARRAY[0,1]'},
        'tool_effect_claims': {"state=ANYARRAY['STARTED','RETURNED_OK','RETURNED_ERROR','UNCONFIRMED','NOT_EXECUTED']"},
    }
    checks = {name: set() for name in profiles}
    primary, foreign = {}, []
    for row in constraints:
        if (row['owner'] not in by_oid or not row['validated'] or not row['enforced'] or not row['local']
                or row['inherited'] or row['parent'] or row['deferred'] or row['deferrable']):
            return False
        name = by_oid[row['owner']]['name']
        kind = row['kind']
        if kind == 'c':
            if not row['builtin_expression'] or row['expression'] is None:
                return False
            program = _check_program(row['expression'])
            if program not in expected_checks[name] or program in checks[name]:
                return False
            checks[name].add(program)
        elif kind == 'p':
            columns = ('scope_digest', 'operation_digest') + (('claim_id',) if name == 'tool_effect_claims' else ())
            if name in primary or row['keys'] != [keys[row['owner']][column] for column in columns]:
                return False
            primary[name] = row
        elif kind == 'f':
            foreign.append(row)
        elif kind == 'n':
            # PostgreSQL 18 also catalogs NOT NULL constraints. Their actual
            # column flags are verified above, and they execute no expression.
            if len(row['keys']) != 1 or row['expression'] is not None:
                return False
        else:
            return False
    if checks != expected_checks or set(primary) != set(profiles) or len(foreign) != 1:
        return False
    parent, child = by_name['tool_effect_operations'], by_name['tool_effect_claims']
    fk = foreign[0]
    pair = ('scope_digest', 'operation_digest')
    if (fk['owner'] != child['oid'] or fk['referenced'] != parent['oid']
            or fk['keys'] != [keys[child['oid']][column] for column in pair]
            or fk['foreign_keys'] != [keys[parent['oid']][column] for column in pair]
            or (fk['update_action'], fk['delete_action'], fk['match']) != ('a', 'a', 's')
            or not fk['builtin_equality'] or fk['index'] != primary['tool_effect_operations']['index']):
        return False
    expected_triggers = {
        (parent['oid'], child['oid'], 'RI_FKey_noaction_del', 9),
        (parent['oid'], child['oid'], 'RI_FKey_noaction_upd', 17),
        (child['oid'], parent['oid'], 'RI_FKey_check_ins', 5),
        (child['oid'], parent['oid'], 'RI_FKey_check_upd', 17),
    }
    observed = set()
    for row in triggers:
        identity = (row['owner'], row['referenced'], row['function'], row['events'])
        if (identity not in expected_triggers or identity in observed
                or not row['internal'] or not row['builtin_function']
                or row['constraint'] != fk['oid'] or row['index'] != fk['index']
                or row['enabled'] != 'O' or row['deferrable'] or row['deferred']
                or row['parent'] or row['nargs'] or row['attributes'] or row['args'] != b''
                or row['when'] is not None or row['old_table'] is not None or row['new_table'] is not None):
            return False
        observed.add(identity)
    return observed == expected_triggers


async def _tool_effect_program_identity(connection, relation):
    """Return immutable, validated catalog identities for BOTH original tables."""
    from .postgres import _POSTGRES_FIRST_NORMAL_OBJECT_ID

    tables = await connection.fetch('''
        SELECT c.oid::bigint AS oid, c.relname AS name,
               ARRAY(SELECT a.attname FROM pg_catalog.pg_attribute a
                 WHERE a.attrelid OPERATOR(pg_catalog.=) c.oid AND a.attnum OPERATOR(pg_catalog.>) 0 AND NOT a.attisdropped ORDER BY a.attnum) AS "columns",
               ARRAY(SELECT a.attnum FROM pg_catalog.pg_attribute a
                 WHERE a.attrelid OPERATOR(pg_catalog.=) c.oid AND a.attnum OPERATOR(pg_catalog.>) 0 AND NOT a.attisdropped ORDER BY a.attnum) AS numbers,
               ARRAY(SELECT a.attnotnull FROM pg_catalog.pg_attribute a
                 WHERE a.attrelid OPERATOR(pg_catalog.=) c.oid AND a.attnum OPERATOR(pg_catalog.>) 0 AND NOT a.attisdropped ORDER BY a.attnum) AS required
        FROM pg_catalog.pg_class c WHERE c.relnamespace OPERATOR(pg_catalog.=) $1::pg_catalog.oid
          AND c.relname OPERATOR(pg_catalog.=) ANY($2::pg_catalog.text[]) ORDER BY c.oid
    ''', relation['namespace_oid'], ['tool_effect_operations', 'tool_effect_claims'])
    ids = [row['oid'] for row in tables]
    if len(ids) != 2 or relation['relation_oid'] not in ids:
        return None
    constraints = await connection.fetch('''
        SELECT k.oid::bigint AS oid, k.conrelid::bigint AS owner, k.contype AS kind,
               k.convalidated AS validated, k.conislocal AS "local", k.coninhcount AS inherited,
               COALESCE((pg_catalog.to_jsonb(k) OPERATOR(pg_catalog.->>) 'conenforced')::pg_catalog.bool, TRUE) AS enforced,
               k.conparentid::bigint AS parent, k.condeferrable AS deferrable, k.condeferred AS deferred,
               k.conkey AS keys, k.confkey AS foreign_keys, k.confrelid::bigint AS referenced,
               k.conindid::bigint AS "index", k.confupdtype AS update_action,
               k.confdeltype AS delete_action, k.confmatchtype AS "match",
               pg_catalog.pg_get_expr(k.conbin, k.conrelid) AS expression,
               NOT EXISTS (SELECT 1 FROM pg_catalog.pg_depend d
                 WHERE d.classid OPERATOR(pg_catalog.=) 'pg_catalog.pg_constraint'::pg_catalog.regclass
                   AND d.objid OPERATOR(pg_catalog.=) k.oid AND d.refobjid OPERATOR(pg_catalog.>=) $2::pg_catalog.oid
                   AND NOT (d.refclassid OPERATOR(pg_catalog.=) 'pg_catalog.pg_class'::pg_catalog.regclass
                     AND d.refobjid OPERATOR(pg_catalog.=) ANY($1::pg_catalog.oid[]))) AS builtin_expression,
               (pg_catalog.cardinality(k.conpfeqop) OPERATOR(pg_catalog.=) 2
                AND pg_catalog.cardinality(k.conppeqop) OPERATOR(pg_catalog.=) 2
                AND pg_catalog.cardinality(k.conffeqop) OPERATOR(pg_catalog.=) 2
                AND NOT EXISTS (SELECT 1 FROM pg_catalog.unnest(k.conpfeqop OPERATOR(pg_catalog.||) k.conppeqop OPERATOR(pg_catalog.||) k.conffeqop) v(oid)
                  LEFT JOIN pg_catalog.pg_operator o ON o.oid OPERATOR(pg_catalog.=) v.oid
                  WHERE o.oid IS NULL OR o.oid OPERATOR(pg_catalog.>=) $2::pg_catalog.oid
                    OR o.oprcode OPERATOR(pg_catalog.<>) pg_catalog.to_regproc('pg_catalog.texteq'))) AS builtin_equality
        FROM pg_catalog.pg_constraint k WHERE k.conrelid OPERATOR(pg_catalog.=) ANY($1::pg_catalog.oid[])
        ORDER BY k.oid
    ''', ids, _POSTGRES_FIRST_NORMAL_OBJECT_ID)
    triggers = await connection.fetch('''
        SELECT t.oid::bigint AS oid, t.tgrelid::bigint AS owner, t.tgconstrrelid::bigint AS referenced,
               t.tgconstraint::bigint AS "constraint", t.tgconstrindid::bigint AS "index",
               p.proname AS "function", t.tgtype AS events, t.tgisinternal AS internal,
               (p.oid OPERATOR(pg_catalog.<) $2::pg_catalog.oid
                AND p.pronamespace OPERATOR(pg_catalog.=) 'pg_catalog'::pg_catalog.regnamespace
                AND l.lanname OPERATOR(pg_catalog.=) 'internal') AS builtin_function,
               t.tgenabled AS enabled, t.tgdeferrable AS deferrable, t.tginitdeferred AS deferred,
               t.tgparentid::bigint AS parent, t.tgnargs AS nargs, t.tgattr::pg_catalog.int2[] AS attributes,
               t.tgargs AS args, t.tgqual::pg_catalog.text AS "when", t.tgoldtable AS old_table, t.tgnewtable AS new_table
        FROM pg_catalog.pg_trigger t
        LEFT JOIN pg_catalog.pg_proc p ON p.oid OPERATOR(pg_catalog.=) t.tgfoid
        LEFT JOIN pg_catalog.pg_language l ON l.oid OPERATOR(pg_catalog.=) p.prolang
        WHERE t.tgrelid OPERATOR(pg_catalog.=) ANY($1::pg_catalog.oid[]) ORDER BY t.oid
    ''', ids, _POSTGRES_FIRST_NORMAL_OBJECT_ID)
    if not _verify_tool_effect_programs(tables, constraints, triggers):
        return None
    def frozen(row):
        return tuple(tuple(value) if type(value) is list else value for value in row.values())
    return tuple(tuple(frozen(row) for row in rows) for rows in (tables, constraints, triggers))
