---
name: cool-then-place
description: Cooling an object before placing it
---

## When to use
The task says to put a *cool* or *cold* or *chilled* object somewhere.

## Procedure
1. Find and `take` the object.
2. `go to fridge 1`.
3. `cool <object> with fridge 1`. One action. Do not open or close the fridge.
4. `go to <target>`, then `put <object> in/on <target>`.

## Common failures
- `open fridge 1` then `put ... in fridge 1`. That stores the object, it does not cool it for the task.
- Cooling at the wrong appliance. Only the fridge cools.
