"""Grid <-> text serialisation for LLM transduction.

Default scheme: one grid row per line, cells written as concatenated digits
(each symbol is a single character 0-9). Compact and unambiguous, e.g.

    012
    345

Parsing is deliberately lenient about model output: we keep only digit
characters per line and drop blank/garbage lines, so a slightly malformed
completion still yields a usable grid (or None if nothing parses).
"""

from __future__ import annotations

from ..io.grid import Grid, MAX_DIM, is_valid_grid

ROW_SEP = "\n"


def grid_to_str(grid: Grid) -> str:
    """Serialise a grid to the canonical compact text form."""
    return ROW_SEP.join("".join(str(c) for c in row) for row in grid)


def str_to_grid(text: str) -> Grid | None:
    """Parse text (possibly noisy model output) back into a Grid, or None.

    Lenient: per line, keep only 0-9 characters; ignore empty lines; stop at the
    first ragged row to avoid swallowing trailing prose. Returns None if the
    result is not a valid grid.
    """
    rows: list[tuple[int, ...]] = []
    width: int | None = None
    for raw_line in text.splitlines():
        digits = [int(ch) for ch in raw_line if ch.isdigit()]
        if not digits:
            # Allow blank separator lines before the grid; once the grid has
            # started, a blank line terminates it.
            if rows:
                break
            continue
        if width is None:
            width = len(digits)
        elif len(digits) != width:
            break  # ragged → end of grid
        rows.append(tuple(digits))
        if len(rows) > MAX_DIM:
            break
    if not rows:
        return None
    grid = tuple(rows)
    return grid if is_valid_grid(grid) else None
