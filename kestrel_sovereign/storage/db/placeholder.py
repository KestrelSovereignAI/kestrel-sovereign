"""
SQL Placeholder Conversion Utilities

Converts between SQLite (?) and PostgreSQL ($1, $2) placeholder styles.
"""

import re
from typing import Tuple


def sqlite_to_postgres(query: str) -> Tuple[str, int]:
    """
    Convert SQLite ? placeholders to PostgreSQL $1, $2, etc.

    Handles:
    - Simple ? placeholders
    - Avoids converting ? inside string literals
    - INSERT OR IGNORE → INSERT ... ON CONFLICT DO NOTHING
    - INSERT OR REPLACE → INSERT ... ON CONFLICT DO UPDATE

    Args:
        query: SQL query with ? placeholders

    Returns:
        Tuple of (converted_query, placeholder_count)

    Example:
        >>> sqlite_to_postgres("SELECT * FROM users WHERE id = ? AND name = ?")
        ('SELECT * FROM users WHERE id = $1 AND name = $2', 2)
    """
    # Check if this is an INSERT OR IGNORE or INSERT OR REPLACE query (before we modify it)
    is_insert_or_ignore = bool(re.search(r'\bINSERT\s+OR\s+IGNORE\s+INTO\b', query, re.IGNORECASE))
    is_insert_or_replace = bool(re.search(r'\bINSERT\s+OR\s+REPLACE\s+INTO\b', query, re.IGNORECASE))

    # Convert SQLite-specific INSERT OR IGNORE to PostgreSQL
    # INSERT OR IGNORE INTO table (...) VALUES (...)
    # → INSERT INTO table (...) VALUES (...) ON CONFLICT DO NOTHING
    query = re.sub(
        r'\bINSERT\s+OR\s+IGNORE\s+INTO\b',
        'INSERT INTO',
        query,
        flags=re.IGNORECASE
    )

    # Convert SQLite-specific INSERT OR REPLACE to PostgreSQL
    # INSERT OR REPLACE INTO table (...) VALUES (...)
    # → INSERT INTO table (...) VALUES (...) ON CONFLICT ... DO UPDATE SET ...
    # Note: This is a simplified conversion - requires the table to have a PRIMARY KEY
    query = re.sub(
        r'\bINSERT\s+OR\s+REPLACE\s+INTO\b',
        'INSERT INTO',
        query,
        flags=re.IGNORECASE
    )

    # State machine to track if we're inside a string literal
    result = []
    count = 0
    i = 0
    in_string = False
    string_char = None
    
    while i < len(query):
        char = query[i]
        
        # Handle string literals (single or double quotes)
        if char in ("'", '"') and not in_string:
            in_string = True
            string_char = char
            result.append(char)
        elif char == string_char and in_string:
            # Check for escaped quotes ('' or "")
            if i + 1 < len(query) and query[i + 1] == string_char:
                result.append(char)
                result.append(char)
                i += 1
            else:
                in_string = False
                string_char = None
                result.append(char)
        elif char == '?' and not in_string:
            count += 1
            result.append(f'${count}')
        else:
            result.append(char)
        
        i += 1

    converted = ''.join(result)

    # Add ON CONFLICT DO NOTHING if this was an INSERT OR IGNORE
    if is_insert_or_ignore:
        converted = converted.rstrip(';').rstrip() + ' ON CONFLICT DO NOTHING'

    # Add ON CONFLICT DO UPDATE if this was an INSERT OR REPLACE
    # This is a simplified conversion that uses EXCLUDED to update all columns
    elif is_insert_or_replace:
        # Extract table name and columns from: INSERT INTO table (col1, col2, ...) VALUES (...)
        match = re.search(
            r'INSERT\s+INTO\s+(\w+)\s*\(([^)]+)\)',
            converted,
            re.IGNORECASE
        )
        if match:
            table_name = match.group(1)
            columns_str = match.group(2)
            columns = [c.strip() for c in columns_str.split(',')]

            # Determine primary key based on known tables
            # graph_nodes: node_id
            # graph_edges: (source_id, target_id, label) composite
            # agent_metadata: (agent_id, key) composite
            known_pks = {
                'graph_nodes': ['node_id'],
                'graph_edges': ['source_id', 'target_id', 'label'],
                'agent_metadata': ['agent_id', 'key'],
            }
            pk_columns = known_pks.get(
                table_name.lower(),
                [columns[0]] if columns else []
            )

            # Build the update clause for non-PK columns
            update_columns = [c for c in columns if c not in pk_columns]
            if update_columns and pk_columns:
                pk_clause = ', '.join(pk_columns)
                update_clause = ', '.join(f'{c} = EXCLUDED.{c}' for c in update_columns)
                converted = converted.rstrip(';').rstrip()
                converted += f' ON CONFLICT ({pk_clause}) DO UPDATE SET {update_clause}'

    return converted, count


def postgres_to_sqlite(query: str) -> str:
    """
    Convert PostgreSQL $1, $2 placeholders to SQLite ? style.
    
    Args:
        query: SQL query with $N placeholders
        
    Returns:
        Query with ? placeholders
        
    Example:
        >>> postgres_to_sqlite("SELECT * FROM users WHERE id = $1 AND name = $2")
        'SELECT * FROM users WHERE id = ? AND name = ?'
    """
    # Replace $N with ? - simple regex since we don't need to preserve order
    return re.sub(r'\$\d+', '?', query)


def normalize_schema(schema: str, backend: str) -> str:
    """
    Normalize CREATE TABLE schema for specific backend.
    
    Conversions:
    - AUTOINCREMENT → SERIAL (postgres)
    - INTEGER PRIMARY KEY → SERIAL PRIMARY KEY (postgres)
    - REAL → DOUBLE PRECISION (postgres)
    - BLOB → BYTEA (postgres)
    - BOOLEAN handling (both support BOOLEAN now)
    
    Args:
        schema: CREATE TABLE statement(s)
        backend: 'sqlite' or 'postgres'
        
    Returns:
        Schema converted for target backend
    """
    if backend == 'sqlite':
        # PostgreSQL → SQLite conversions
        schema = re.sub(r'\bSERIAL\b', 'INTEGER', schema, flags=re.IGNORECASE)
        schema = re.sub(r'\bBIGSERIAL\b', 'INTEGER', schema, flags=re.IGNORECASE)
        schema = re.sub(r'\bDOUBLE PRECISION\b', 'REAL', schema, flags=re.IGNORECASE)
        schema = re.sub(r'\bBYTEA\b', 'BLOB', schema, flags=re.IGNORECASE)
        schema = re.sub(r'\bTIMESTAMP WITH TIME ZONE\b', 'TEXT', schema, flags=re.IGNORECASE)
        schema = re.sub(r'\bTIMESTAMPTZ\b', 'TEXT', schema, flags=re.IGNORECASE)
        schema = re.sub(r'\bJSONB?\b', 'TEXT', schema, flags=re.IGNORECASE)
        # Remove PostgreSQL-specific clauses
        schema = re.sub(r'\s+DEFAULT\s+NOW\(\)', '', schema, flags=re.IGNORECASE)
        schema = re.sub(r'\s+DEFAULT\s+CURRENT_TIMESTAMP', '', schema, flags=re.IGNORECASE)
        return schema
    
    elif backend == 'postgres':
        # SQLite → PostgreSQL conversions
        schema = re.sub(
            r'INTEGER\s+PRIMARY\s+KEY\s+AUTOINCREMENT',
            'SERIAL PRIMARY KEY',
            schema,
            flags=re.IGNORECASE
        )
        schema = re.sub(
            r'INTEGER\s+PRIMARY\s+KEY',
            'SERIAL PRIMARY KEY',
            schema,
            flags=re.IGNORECASE
        )
        schema = re.sub(r'\bAUTOINCREMENT\b', '', schema, flags=re.IGNORECASE)
        schema = re.sub(r'\bREAL\b', 'DOUBLE PRECISION', schema, flags=re.IGNORECASE)
        schema = re.sub(r'\bBLOB\b', 'BYTEA', schema, flags=re.IGNORECASE)
        return schema
    
    return schema
