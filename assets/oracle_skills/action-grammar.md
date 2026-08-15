---
name: action-grammar
description: The exact action grammar the environment accepts
---

## Why this matters
The environment accepts a small fixed set of action forms. Anything else is rejected and wastes a turn.

## The complete grammar
- `go to <receptacle>`
- `open <receptacle>` / `close <receptacle>`
- `take <object> from <receptacle>`
- `put <object> in/on <receptacle>`
- `clean <object> with <receptacle>`
- `heat <object> with <receptacle>`
- `cool <object> with <receptacle>`
- `use <object>`
- `examine <object|receptacle>`
- `look`

## Rules
- Always include the numeric index: `cabinet 4`, not `cabinet`.
- One action per turn, with no explanation attached to it.
- You must be at a receptacle to interact with it, and holding an object to transform it.
- If an action is rejected, the wording was wrong. Re-read the admissible list rather than repeating it.
