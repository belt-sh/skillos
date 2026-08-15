---
name: pick-and-place
description: Moving one object to a receptacle
---

## When to use
The task names one object and one destination and asks for no transformation.

## Procedure
1. Search likely locations for the object. Go to a receptacle before inspecting it.
2. `take <object> from <receptacle>`.
3. `go to <target receptacle>`.
4. `put <object> in/on <target receptacle>`.

## Common failures
- `take` from across the room. You must `go to` the receptacle first.
- Opening a closed receptacle is required before its contents are visible: `open <receptacle>`.
- Naming the object without its index. Use the exact name shown, for example `mug 2`.
