from __future__ import annotations

import html
import io
from dataclasses import dataclass
from typing import List, Optional, Literal, Iterator
from contextlib import contextmanager, redirect_stdout, redirect_stderr

import ipywidgets as w


Level = Literal["info", "warning", "error"]


@dataclass
class _Entry:
    level: Level
    message: str
    count: int = 1


class HtmlLog:
    """
    Notebook-stable log view based on a single HTML widget.

    Why this exists:
      Some notebook front-ends (notably VS Code) can duplicate ipywidgets.Output rendering,
      causing a single print to appear multiple times. This logger avoids Output capture and
      renders the log as HTML, which is stable across front-ends.

    Features:
      - severity coloring: warnings in orange, errors in red
      - coalescing of consecutive identical messages (shows xN)
      - bounded history (drops oldest entries beyond max_entries)
      - optional "Output-like" capture proxy to minimize code changes
    """

    def __init__(self, *, title: str | None = None, height_px: int = 220, max_entries: int = 2000) -> None:
        self._entries: List[_Entry] = []
        self._height_px = int(height_px)
        self._max_entries = int(max_entries)
        self.widget = w.HTML()
        # Convenience container used by the GUI (some panels expect .panel).
        if title:
            self.panel = w.VBox([w.HTML(f"<b>{html.escape(str(title))}</b>"), self.widget])
        else:
            self.panel = self.widget
        self.clear()

    def clear(self) -> None:
        self._entries.clear()
        self._render()

    def info(self, message: str) -> None:
        self._add("info", message)

    def warning(self, message: str) -> None:
        self._add("warning", message)

    def error(self, message: str) -> None:
        self._add("error", message)

    def write(self, message: str) -> None:
        """Compatibility helper for older GUI code.

        The Phase III GUI uses HtmlLog.write(...) and may pass HTML snippets.
        We strip basic HTML tags and route each line through the severity classifier.
        """
        import re
        txt = "" if message is None else str(message)
        plain = re.sub(r"<[^>]+>", "", txt)
        lines = plain.splitlines() or [""]
        for line in lines:
            level = self._classify(line)
            self._add(level, line)

    def output_proxy(self) -> "_OutputProxy":
        """
        Return an Output-like proxy usable as:

            out_log = log.output_proxy()
            with out_log:
                print("...")

        The proxy captures stdout/stderr and routes each printed line into this HtmlLog
        with automatic severity classification.
        """
        return _OutputProxy(self)

    # -------------------------
    # Internals
    # -------------------------
    def _classify(self, line: str) -> Level:
        s = (line or "").lstrip()
        if s.startswith(("ERROR:", "Error:", "Exception:", "Traceback")):
            return "error"
        if s.startswith(("WARNING:", "Warning:", "WARN", "CHECK:")):
            return "warning"
        return "info"

    def _add(self, level: Level, message: str) -> None:
        msg = "" if message is None else str(message)

        # Coalesce consecutive duplicates
        if self._entries and self._entries[-1].level == level and self._entries[-1].message == msg:
            self._entries[-1].count += 1
            self._render()
            return

        self._entries.append(_Entry(level=level, message=msg, count=1))
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries :]
        self._render()

    def _render(self) -> None:
        def color(level: Level) -> str:
            if level == "error":
                return "#b00020"  # red
            if level == "warning":
                return "#b26a00"  # orange
            return "#222222"     # near-black

        rows = []
        for e in self._entries:
            txt = html.escape(e.message)
            suffix = f" (x{e.count})" if e.count > 1 else ""
            rows.append(
                f"<div style='color:{color(e.level)}; white-space:pre-wrap; "
                f"font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, Liberation Mono, Courier New, monospace;'>{txt}{html.escape(suffix)}</div>"
            )

        inner = "".join(rows) if rows else "<div style='color:#666;'>Log is empty.</div>"
        self.widget.value = (
            f"<div style='border:1px solid #ddd; padding:8px; height:{self._height_px}px; "
            f"overflow-y:auto; background:#fff;'>{inner}</div>"
        )


class _OutputProxy:
    """
    Minimal ipywidgets.Output-like proxy:
      - context manager capturing stdout/stderr
      - clear_output() maps to HtmlLog.clear()
    """

    def __init__(self, log: HtmlLog) -> None:
        self._log = log
        self._buf_out: Optional[io.StringIO] = None
        self._buf_err: Optional[io.StringIO] = None
        self._cm_out = None
        self._cm_err = None

    def clear_output(self, wait: bool = False) -> None:
        _ = wait
        self._log.clear()

    def __enter__(self) -> "_OutputProxy":
        self._buf_out = io.StringIO()
        self._buf_err = io.StringIO()
        self._cm_out = redirect_stdout(self._buf_out)
        self._cm_err = redirect_stderr(self._buf_err)
        self._cm_out.__enter__()
        self._cm_err.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        # Stop redirects first
        try:
            if self._cm_err is not None:
                self._cm_err.__exit__(exc_type, exc, tb)
        finally:
            if self._cm_out is not None:
                self._cm_out.__exit__(exc_type, exc, tb)

        out = self._buf_out.getvalue() if self._buf_out is not None else ""
        err = self._buf_err.getvalue() if self._buf_err is not None else ""

        text = ""
        if out:
            text += out
        if err:
            text += err

        for line in (text or "").splitlines():
            lvl = self._log._classify(line)
            if lvl == "error":
                self._log.error(line)
            elif lvl == "warning":
                self._log.warning(line)
            else:
                self._log.info(line)

        # Do not suppress exceptions
        return False
