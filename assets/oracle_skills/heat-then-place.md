---
name: heat-then-place
description: Heating an object before placing it
---

## When to use
The task says to put a *hot* or *heated* object somewhere.

## Procedure
1. Find and `take` the object.
2. `go to microwave 1`.
3. `heat <object> with microwave 1`. One action. Do not open, insert, set a time, or start it.
4. `go to <target>`, then `put <object> in/on <target>`.

## Common failures
- Opening the microwave first. `heat X with microwave 1` handles the whole interaction.
- Using a stoveburner when a microwave is present. Prefer the microwave.
- Heating before picking the object up.
