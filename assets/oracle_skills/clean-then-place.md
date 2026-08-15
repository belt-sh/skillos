---
name: clean-then-place
description: Cleaning an object before placing it
---

## When to use
The task says to put a *clean* object somewhere.

## Procedure
1. Find the object. Check countertops, tables, shelves, cabinets and the sink first.
2. `take <object> from <receptacle>` while standing at that receptacle.
3. `go to sinkbasin 1`.
4. `clean <object> with sinkbasin 1`. This is one action. Do not turn on a faucet.
5. `go to <target receptacle>`, then `put <object> in/on <target>`.

## Common failures
- Issuing `clean` before you are at the sink. You must `go to sinkbasin 1` first.
- Trying to wash with an invented verb. The only cleaning verb is `clean X with Y`.
- Forgetting that you must be holding the object before cleaning it.
