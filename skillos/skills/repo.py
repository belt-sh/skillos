"""Markdown skill repository with BM25 retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import yaml
from rank_bm25 import BM25Okapi


@dataclass
class Skill:
    name: str
    description: str
    content: str  # full markdown including frontmatter

    @classmethod
    def from_markdown(cls, markdown: str) -> Skill:
        """Parse a skill from markdown with YAML frontmatter."""
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", markdown, re.DOTALL)
        if not match:
            raise ValueError("Skill must have YAML frontmatter delimited by ---")
        meta = yaml.safe_load(match.group(1))
        return cls(
            name=meta["name"],
            description=meta.get("description", ""),
            content=markdown,
        )

    def searchable_text(self) -> str:
        return f"{self.name} {self.description} {self.content}"


class SkillRepo:
    """In-memory skill repository with BM25 retrieval.

    This is the SkillRepo from the paper — skills as markdown files,
    managed via insert/update/delete operations, retrieved via BM25.
    """

    def __init__(self):
        self.skills: dict[str, Skill] = {}
        self._bm25: BM25Okapi | None = None
        self._bm25_dirty = True

    def insert(self, skill_name: str, content: str) -> bool:
        """Insert a new skill. Returns True if successful."""
        if skill_name in self.skills:
            return False
        try:
            skill = Skill.from_markdown(content)
            # Use the provided name, not the one parsed from frontmatter
            skill.name = skill_name
            self.skills[skill_name] = skill
            self._bm25_dirty = True
            return True
        except (ValueError, yaml.YAMLError):
            return False

    def update(self, skill_name: str, new_name: str | None = None, new_content: str | None = None) -> bool:
        """Update an existing skill. Returns True if successful."""
        if skill_name not in self.skills:
            return False
        skill = self.skills[skill_name]
        if new_content is not None:
            try:
                updated = Skill.from_markdown(new_content)
                skill.content = new_content
                skill.description = updated.description
            except (ValueError, yaml.YAMLError):
                return False
        if new_name is not None and new_name != skill_name:
            skill.name = new_name
            self.skills[new_name] = skill
            del self.skills[skill_name]
        self._bm25_dirty = True
        return True

    def delete(self, skill_name: str) -> bool:
        """Delete a skill. Returns True if successful."""
        if skill_name not in self.skills:
            return False
        del self.skills[skill_name]
        self._bm25_dirty = True
        return True

    def replace_skills(self, skills: dict[str, Skill]) -> None:
        """Replace the entire skill set (used when restoring from a checkpoint)
        and invalidate the BM25 index so the next retrieve() rebuilds it."""
        self.skills = skills
        self._bm25_dirty = True

    def retrieve(self, query: str, top_k: int = 5) -> list[Skill]:
        """Retrieve top-k skills by BM25 relevance."""
        if not self.skills:
            return []
        if self._bm25_dirty:
            self._rebuild_index()
        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)
        skill_list = list(self.skills.values())
        ranked = sorted(zip(scores, skill_list), key=lambda x: x[0], reverse=True)
        return [skill for _, skill in ranked[:top_k]]

    def format_skills(self, skills: list[Skill]) -> str:
        """Format skills for injection into a prompt."""
        if not skills:
            return "No relevant skills found."
        parts = []
        for i, skill in enumerate(skills, 1):
            parts.append(f"### Skill {i}: {skill.name}\n{skill.content}")
        return "\n\n".join(parts)

    def total_tokens(self) -> int:
        """Approximate token count of entire repo (for compression reward)."""
        return sum(len(s.content.split()) for s in self.skills.values())

    def clone(self) -> SkillRepo:
        """Create a deep copy of this repo."""
        new = SkillRepo()
        for name, skill in self.skills.items():
            new.skills[name] = Skill(
                name=skill.name,
                description=skill.description,
                content=skill.content,
            )
        new._bm25_dirty = True
        return new

    def save(self, directory: str) -> None:
        """Persist every skill as a `<name>.md` file under `directory/`.

        Mirror of how `belt` and Anthropic SKILL.md format store skills on
        disk — one markdown file per skill, full original content (with
        YAML frontmatter) preserved.
        """
        import os
        os.makedirs(directory, exist_ok=True)
        # Stale skills could linger if names changed; clean existing .md first.
        for fn in os.listdir(directory):
            if fn.endswith(".md"):
                os.remove(os.path.join(directory, fn))
        for name, skill in self.skills.items():
            # Sanitize the filename — skill names can contain anything the
            # curator emits.
            safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)[:200]
            with open(os.path.join(directory, f"{safe}.md"), "w") as f:
                f.write(skill.content)

    @classmethod
    def load(cls, directory: str) -> SkillRepo:
        """Load every `*.md` file under `directory/` as a skill.

        Skill names come from the YAML frontmatter (canonical), not the
        filename. Files that fail to parse are skipped quietly.
        """
        import os
        repo = cls()
        if not os.path.isdir(directory):
            return repo
        for fn in sorted(os.listdir(directory)):
            if not fn.endswith(".md"):
                continue
            try:
                with open(os.path.join(directory, fn)) as f:
                    content = f.read()
                skill = Skill.from_markdown(content)
                repo.skills[skill.name] = skill
            except Exception:
                continue
        repo._bm25_dirty = True
        return repo

    def _rebuild_index(self):
        corpus = [s.searchable_text().lower().split() for s in self.skills.values()]
        if corpus:
            self._bm25 = BM25Okapi(corpus)
        self._bm25_dirty = False

    def __len__(self) -> int:
        return len(self.skills)
