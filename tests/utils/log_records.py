"""Find caller content in captured log records, whatever field carries it.

A record can carry text in its format string, its arguments, an ``extra``
attribute, its exception, or its stack. Checking the rendered message alone
misses all but the first, and checking for one known log line misses every
other line, so this looks at all of them (#3318).
"""

import logging
from typing import Iterable, List


def _everything_in(record: logging.LogRecord) -> str:
    # ``vars`` holds the format string and its arguments unrendered, so text
    # passed to a call whose arguments do not fit its format is still seen.
    parts = [repr(vars(record))]
    try:
        parts.append(record.getMessage())
    except (TypeError, ValueError):
        pass
    if record.exc_info:
        parts.append(logging.Formatter().formatException(record.exc_info))
    return "\n".join(parts)


def records_containing(
    records: Iterable[logging.LogRecord], needle: str
) -> List[logging.LogRecord]:
    """Return every record that carries ``needle`` anywhere."""
    return [record for record in records if needle in _everything_in(record)]
