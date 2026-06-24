"""Shared native (tkinter) file/folder dialog helpers for the GUI.

Centralises the tkinter root lifecycle (create -> withdraw -> raise topmost ->
destroy) that was previously duplicated across every GUI tab module. Each
function returns ``None`` when tkinter is unavailable (e.g. headless) or when
the user cancels the dialog.
"""

from __future__ import annotations

import contextlib
from typing import Callable, List, Optional, Tuple


def _with_root(action: "Callable[[object], Optional[str]]") -> Optional[str]:
    """Run *action(filedialog)* inside a managed, hidden tkinter root.

    Returns ``None`` if tkinter cannot be imported; always destroys the root.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None

    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        with contextlib.suppress(Exception):
            root.attributes("-topmost", True)
        return action(filedialog)
    finally:
        with contextlib.suppress(Exception):
            if root is not None:
                root.destroy()


def browse_for_folder(title: str = "Select folder") -> Optional[str]:
    """Native folder chooser. Returns the path, or None if cancelled/unavailable."""
    def _act(filedialog):
        p = filedialog.askdirectory(title=title)
        return str(p) if p else None

    return _with_root(_act)


def open_file_dialog(
    *,
    title: str = "Select file",
    filetypes: Optional[List[Tuple[str, str]]] = None,
) -> Optional[str]:
    """Native Open dialog. Returns the path, or None if cancelled/unavailable."""
    def _act(filedialog):
        p = filedialog.askopenfilename(
            title=title,
            filetypes=filetypes or [("All files", "*.*")],
        )
        return str(p) if p else None

    return _with_root(_act)


def saveas_dialog(
    *,
    initialfile: str = "",
    defaultextension: str = "",
    filetypes: Optional[List[Tuple[str, str]]] = None,
    title: str = "Save file",
    initialdir: Optional[str] = None,
) -> Optional[str]:
    """Native Save-As dialog. Returns the path, or None if cancelled/unavailable."""
    def _act(filedialog):
        kwargs = dict(
            title=title,
            initialfile=initialfile,
            defaultextension=defaultextension,
            filetypes=filetypes or [("All files", "*.*")],
        )
        if initialdir is not None:
            kwargs["initialdir"] = initialdir
        p = filedialog.asksaveasfilename(**kwargs)
        return str(p) if p else None

    return _with_root(_act)
