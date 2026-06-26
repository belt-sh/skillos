"""Canonical ALFWorld task-type taxonomy and classifiers.

One home for the 6-type partition (paper §3.2.1) so the training env and the
eval harness can never drift. Two inputs are supported: the gamefile path
(reliable, no env run) and the task description (used when only the description
is in hand). Both return a canonical lowercase label from ``TASK_TYPES``;
``DISPLAY`` maps to the Table-1 capitalized form for reporting.
"""

from __future__ import annotations

# Canonical lowercase labels.
TASK_TYPES = ("pick", "look", "clean", "heat", "cool", "pick2")

# Capitalized forms for eval reporting (paper Table 1 column names).
DISPLAY = {"pick": "Pick", "look": "Look", "clean": "Clean",
           "heat": "Heat", "cool": "Cool", "pick2": "Pick2"}


def classify_gamefile(gamefile: str) -> str:
    """Task type from the ALFWorld gamefile path (reliable, no env run)."""
    g = (gamefile or "").lower()
    if "pick_two" in g:
        return "pick2"
    if "pick_clean" in g:
        return "clean"
    if "pick_heat" in g:
        return "heat"
    if "pick_cool" in g:
        return "cool"
    if "look_at" in g:
        return "look"
    return "pick"


def classify_description(description: str) -> str:
    """Task type from an ALFWorld task description.

    Order matters: 'pick2' (two) before the transform verbs, which precede the
    generic pick/look fallbacks.
    """
    d = (description or "").lower()
    if "two" in d:
        return "pick2"
    if "clean" in d:
        return "clean"
    if "hot" in d or "heat" in d:
        return "heat"
    if "cool" in d:
        return "cool"
    if ("look" in d or "examine" in d) and "lamp" in d:
        return "look"
    return "pick"
