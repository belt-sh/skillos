"""Curator model — applies insert/update/delete operations from tool calls.

The curator is the model being trained. It receives a trajectory + existing skills
and outputs tool calls (new_skill_insert, skill_update, skill_delete).

This module handles parsing the model's tool call output and applying operations
to the SkillRepo.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from skillos.skills.repo import SkillRepo


@dataclass
class CurationOp:
    """A single curation operation parsed from model output."""
    name: str  # new_skill_insert, skill_update, skill_delete
    arguments: dict
    valid: bool = True
    executed: bool = False


def parse_tool_calls(model_output: str) -> list[CurationOp]:
    """Parse tool calls from model output.

    Handles both structured tool call format (from TRL) and
    JSON-in-text format that models sometimes produce.
    """
    ops = []

    # Try to find function call JSON blocks
    # Pattern: {"name": "...", "arguments": {...}}
    for match in re.finditer(
        r'\{\s*"name"\s*:\s*"(new_skill_insert|skill_update|skill_delete)".*?\}(?=\s*(?:\{|$))',
        model_output,
        re.DOTALL,
    ):
        try:
            call = json.loads(match.group())
            ops.append(CurationOp(
                name=call["name"],
                arguments=call.get("arguments", {}),
            ))
        except json.JSONDecodeError:
            ops.append(CurationOp(
                name="unknown",
                arguments={},
                valid=False,
            ))

    return ops


def apply_curation_ops(repo: SkillRepo, ops: list[CurationOp]) -> list[CurationOp]:
    """Apply a list of curation operations to a skill repo.

    Returns the ops with executed=True/False set.
    """
    for op in ops:
        if op.name == "new_skill_insert":
            skill_name = op.arguments.get("skill_name", "")
            content = op.arguments.get("content", "")
            if skill_name and content:
                op.executed = repo.insert(skill_name, content)
            else:
                op.valid = False
                op.executed = False

        elif op.name == "skill_update":
            skill_name = op.arguments.get("skill_name", "")
            new_name = op.arguments.get("new_name")
            new_content = op.arguments.get("new_content")
            if skill_name:
                op.executed = repo.update(skill_name, new_name, new_content)
            else:
                op.valid = False
                op.executed = False

        elif op.name == "skill_delete":
            skill_name = op.arguments.get("skill_name", "")
            if skill_name:
                op.executed = repo.delete(skill_name)
            else:
                op.valid = False
                op.executed = False

        else:
            op.valid = False
            op.executed = False

    return ops
