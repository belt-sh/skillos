---
name: search-order
description: Where to look for things
---

## Why this matters
Most failures are search failures, not manipulation failures. The object is findable; the agent runs out of turns looking.

## Typical locations
- Utensils, mugs, cups, bowls, plates: countertop, sink, cabinet, drawer, shelf.
- Food: fridge, countertop, microwave, garbagecan.
- Books, pens, laptops, keys, watches: desk, sidetable, shelf, drawer, bed, sofa.
- Cloth, towels, soap, toiletpaper: countertop, toilet, sink, cabinet, shelf, handtowelholder.
- Statues, vases, boxes: shelf, sidetable, dresser, coffeetable.

## Rules
- Sweep open surfaces first, they need no `open`. Then move to closed containers.
- Do not revisit a receptacle you have already examined this episode.
- Closed receptacles need `open <receptacle>` before their contents appear.
