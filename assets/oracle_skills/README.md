# Author-written oracle skill repository

An upper-bound control for the curation experiments. These skills were written
directly (not by a trained curator, and not from experience) from the
**published ALFWorld task-type definitions and action grammar only**.

**Authorship, stated precisely because the result depends on it:** they were
written by the LLM agent conducting this study, not by a human. Calling them
"hand-written" would overstate the comparison. The honest description is
"written by a capable model with documentation access and no eval data", which
is a different and easier condition than the curator's (which must infer the
same content from rollouts) but is not a human baseline.

They were not derived from, tuned on, or checked against any evaluation episode,
transcript, or per-game result.

Why it exists: every curator arm in this study is compared against "no notes."
That answers "does this curator help" but not "is there anything useful a note
could say here." If competently written notes also fail to lift the executor,
then the null results are a property of the executor and the benchmark, not of
GRPO curation. If directly-authored notes lift it substantially, then the ceiling is real
and the trained curators are simply not reaching it.

Eight skills, one per ALFWorld task type plus two general ones, matching the
scale of a mid-training curator repository (typical trained repos in this study
hold 6 to 20 skills).

Provenance note for the paper: written 2026-08-15, before the oracle arm was
run, and committed unchanged. The commit that adds this directory precedes the
commit that adds the arm's results.
