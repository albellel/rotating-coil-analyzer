"""Validation utilities.

This package contains *non-interactive* tooling intended for golden-standard
validation campaigns.

Design goals
------------
1) Keep validation code out of the GUI path (no UI coupling).
2) Make comparisons reproducible and scriptable (CLI-style entry points).
3) Prefer robust parsing of legacy/reference exports.
"""
