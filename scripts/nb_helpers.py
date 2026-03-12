"""Helpers for programmatic notebook construction.

Handles the three recurring pitfalls on Windows + Git Bash:

1. **Newlines in f-strings**: ``code(r"ax.set_ylabel(f'{x}\\nY')")`` — use raw
   strings so ``\\n`` stays as two characters in the notebook source.
   If you forget, ``validate_source()`` catches it.

2. **Unicode / cp1252**: All notebook JSON is written with ``ensure_ascii=False``
   and ``encoding="utf-8"``.  Print helpers reconfigure stdout.

3. **Smart quotes**: ``sanitize()`` replaces ``\u2018 \u2019 \u201c \u201d``
   with ASCII equivalents in code cells (keeps them in markdown).

Usage
-----
>>> from nb_helpers import code, md, write_notebook
>>> cells = [
...     md("title", "# My Notebook"),
...     code("setup", r'''
...         import numpy as np
...         ax.set_ylabel(SEG_LABEL[seg] + '\\nB1 (T)')
...     '''),
... ]
>>> write_notebook(Path("my_notebook.ipynb"), cells)

The ``code()`` function:
- Accepts a **single multiline string** (triple-quoted)
- Auto-dedents (strips common leading whitespace)
- Strips one leading blank line (from triple-quote style)
- Validates: no real newlines inside unclosed f-strings
- Splits into notebook source lines (each ending with ``\\n`` except the last)
"""
from __future__ import annotations

import json
import re
import sys
import textwrap
from pathlib import Path


def _to_source_lines(text: str) -> list[str]:
    """Split text into notebook source lines."""
    lines = text.split("\n")
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


def _validate_source(source_lines: list[str], cell_id: str) -> None:
    """Check for common mistakes in code cell source.

    Raises ValueError if:
    - An f-string literal spans multiple source lines (real newline inside f"...")
    - Smart quotes appear in code
    """
    for i, line in enumerate(source_lines):
        stripped = line.rstrip("\n")

        # Detect unclosed f-string: a line opens f"..." or f'...' but
        # doesn't close it (the closing quote is missing or on the next line).
        # We check: after the f-quote opener, is there a matching close quote?
        for quote in ['"', "'"]:
            marker = f"f{quote}"
            idx = stripped.find(marker)
            if idx == -1:
                continue
            # Walk from after the opening quote to find the closing one,
            # skipping escaped quotes and {..} expressions
            rest = stripped[idx + 2:]
            depth = 0
            closed = False
            j = 0
            while j < len(rest):
                ch = rest[j]
                if ch == "{":
                    depth += 1
                elif ch == "}" and depth > 0:
                    depth -= 1
                elif ch == "\\" and j + 1 < len(rest):
                    j += 1  # skip escaped char
                elif ch == quote and depth == 0:
                    closed = True
                    break
                j += 1
            if not closed:
                raise ValueError(
                    f"Cell {cell_id!r}, line {i+1}: unclosed f-string "
                    f"(real newline inside f-string?):\n  {stripped}"
                )

        # Smart quotes in code
        for bad, good in [("\u2018", "'"), ("\u2019", "'"),
                          ("\u201c", '"'), ("\u201d", '"')]:
            if bad in stripped:
                raise ValueError(
                    f"Cell {cell_id!r}, line {i+1}: smart quote {bad!r} found. "
                    f"Replace with {good!r}:\n  {stripped}"
                )


def _sanitize_smart_quotes(text: str) -> str:
    """Replace smart quotes with ASCII equivalents."""
    return (text
            .replace("\u2018", "'").replace("\u2019", "'")
            .replace("\u201c", '"').replace("\u201d", '"'))


def code(cell_id: str, text: str, *, validate: bool = True) -> dict:
    """Create a code cell from a multiline string.

    Use raw strings (r'''...''') to preserve backslash-n as two characters.
    The text is auto-dedented and leading/trailing blank lines are stripped.
    """
    text = textwrap.dedent(text).strip("\n")
    text = _sanitize_smart_quotes(text)
    source = _to_source_lines(text)
    if validate:
        _validate_source(source, cell_id)
    return {
        "cell_type": "code",
        "id": cell_id,
        "metadata": {},
        "source": source,
        "outputs": [],
        "execution_count": None,
    }


def md(cell_id: str, text: str) -> dict:
    """Create a markdown cell from a multiline string.

    Auto-dedented; leading/trailing blank lines stripped.
    Unicode (em-dash, mu, sigma) is fine in markdown.
    """
    text = textwrap.dedent(text).strip("\n")
    source = _to_source_lines(text)
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": source,
    }


def write_notebook(path: Path, cells: list[dict]) -> None:
    """Write a notebook to disk (UTF-8, no ASCII escaping)."""
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.13.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"  Wrote {path} ({len(cells)} cells)")


def patch_cell_source(nb_path: Path, cell_id: str, new_text: str,
                      *, validate: bool = True) -> None:
    """Replace the source of a single cell in an existing notebook.

    Reads the notebook, finds the cell by ID, replaces its source,
    clears outputs, and writes back.
    """
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    new_text = textwrap.dedent(new_text).strip("\n")
    new_text = _sanitize_smart_quotes(new_text)
    source = _to_source_lines(new_text)
    if validate:
        _validate_source(source, cell_id)

    found = False
    for cell in nb["cells"]:
        if cell.get("id") == cell_id:
            cell["source"] = source
            if cell["cell_type"] == "code":
                cell["outputs"] = []
                cell["execution_count"] = None
            found = True
            break

    if not found:
        raise KeyError(f"Cell {cell_id!r} not found in {nb_path}")

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)


def safe_print(*args, **kwargs):
    """Print with UTF-8 stdout (Windows cp1252 workaround)."""
    if sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(*args, **kwargs)
