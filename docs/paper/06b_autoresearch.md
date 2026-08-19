# 6.8 Conduct of an agent-run reproduction

This study was carried out almost entirely by an LLM agent working continuously
for roughly three months, with a human author setting scope, funding compute, and
adjudicating disputes. That is unusual enough to be worth reporting as a result in
its own right.

**This section is about the agent's research behaviour, not about SkillOS.** The
scientific findings are in Section 5 and are evaluated there on their own
evidence. Here we ask a different question: as a way of doing research, what did
this mode of work get wrong, what did it get right, and what should be gated by a
human. We report both halves because both are true and only one of them normally
survives into a paper.

We are deliberately concrete. A general claim that "agents make mistakes" is not
useful. A count, a taxonomy, and a cost are. Where a behaviour had a scientific
consequence we cite the relevant result in Section 5 as evidence, rather than
claiming the result as a contribution of the method of work.

## 6.8.1 How the failures were counted

The study ran as one continuous session: a 79 MB transcript, 18,409
conversational turns. At the end we compressed it to prose and had eight
independent reviewer agents each audit one contiguous slice, cross-checking every
failure they found against the four records the project had been keeping (a
running journal, a divergences ledger, a results report, and one postmortem).
Each failure was labelled DOCUMENTED, PARTIAL, or MISSING.

The audit was not self-assessment: the reviewers had the transcript, not the
memory of it, and no access to each other's output. **It is also not
triangulation, and an earlier draft of this section overstated it.** Two slices
overlapped in time and at least seven incidents were found twice, which we first
described as independent cross-confirmation. A verification pass rejected that
framing, correctly. Both reviewers read overlapping ranges of the *same*
mechanical digest of the *same* transcript, so their agreement demonstrates that
the extraction is reproducible across reader contexts, not that the extraction is
true. Every figure they agreed on is a literal string in the source.

The one genuinely useful thing the overlap produced was a disagreement, and the
reconciliation we published for it was wrong. One reviewer put a lost-training
incident at ~15 hours and the other at ~22, which we reported as a magnitude
dispute. It was not: 14.6 hours of completed training time sat inside a 21h55m
wall, the second reviewer had labelled its figure "wall", and our summary erased
the qualifier to manufacture a conflict. **Both readers were right and the
synthesis was wrong**, which argues for checking figures against timestamps
rather than for trusting reviewer consensus.

Therefore a second pass re-verified every row against the raw transcript, the
installed libraries and git history, with instructions to default to UNSUPPORTED.
That pass is the reason the numbers below are stated as they are, and it changed
several: see §6.8.6.

The full ledger is released with the artifact. The counts:

| | entries | share |
|---|---|---|
| Failures found | 185 | |
| Documented in the project's own records | 36 | 19% |
| Recorded, but without cause, cost, or trigger (PARTIAL) | 55 | 30% |
| **Absent from every record (MISSING)** | **93** | **50%** |

**These integers are approximate and should be read as such.** Verification found
at least one incident entered twice under different dates, so the total is
inflated by an unknown small number; the digest also truncates long tool outputs,
so failures visible only inside a large log are missed entirely. The defensible
claim is the ratio, not the count: **roughly half of what happened was absent
from the records, and about a fifth was documented with its cause.** We report the
raw integers only because withholding them would make the ratio unauditable.

## 6.8.2 The shortcomings

Fourteen recurring modes emerged. We report the six with the largest
consequences for the science, since the operational ones (self-inflicted process
kills, orphaned monitors, and idle GPU time across at least seven incidents of
which only one, a 44-hour stall, is precisely established) cost money rather than
validity.

**A sentinel that does not say why.** At least eight code paths substituted a
plausible value for a measurement that never happened: an unparseable action
became the first admissible action, an episode that never ran became
`success: False`, a rollout with no measured position became `r_task = 0.0`, an
API failure became a wrong answer. None carried a flag distinguishing "measured
and failed" from "never measured".

This is the single most important methodological point in this paper, because of
its statistical consequence. **A systematically crippled arm is reliably
crippled.** Low variance in a consistent direction is exactly what a paired
significance test rewards. Our most significant result, at p below 0.0005, was an
eight-hour authentication outage measured against a healthy control. It was not a
weak effect that squeaked past a threshold; it was a strong, clean, reproducible
measurement of a broken pipe. Agent-run work is unusually exposed to this because
the agent writes both the harness and the analysis, and a defensive default that
keeps a long job alive is locally the right call at 3 a.m. and globally fatal to
the result.

**Health metrics validated against each other, never against ground truth.**
Three consecutive runs, two of them multi-day and about eight *box*-days on eight
H100s (~64 GPU-days), trained on ten fixed tasks because an environment
identifier was always zero. Reward, KL, gradient
norm, and a purpose-built degeneracy tripwire all looked healthy throughout. The
tripwire asserted that reward must vary within a GRPO group, and a second bug,
per-rank seed divergence, satisfied it. A check that asserts the absence of a
symptom can be satisfied by the next defect. What worked, every time, was
printing the quantity that should be invariant: measured positions per rollout,
distinct tasks drawn, coercion rate.

**A dependency's comment taken as its behaviour.** The training framework applies
its completion-length cap to the *accumulated* multi-turn completion, not to each
response. At the paper's 4096 tokens, a ten-position rollout carrying ten
trajectories fits about three positions, after which the framework silently drops
the tool result. Every training run in this project, for eleven weeks, trained on
roughly three positions of a ten-position protocol. Nothing errored; a truncated
rollout is byte-identical in shape to a finished one. The value looked
intentional because a comment beside it named the paper's hyperparameter. The same
mode produced a false belief about rollout concurrency that cost a factor of four
in wall-clock on every run.

**Fixes sized to the symptom rather than derived from arithmetic.** An
out-of-memory failure requesting 4.64 GiB was answered by offloading optimizer
state, freeing about 8 GiB. The next attempt requested 14.21 GiB, which is the
logits tensor, `per_device x sequence x vocabulary`, computable before the first
launch. Roughly four GPU-hours to learn that the fix had been aimed at the wrong
tensor.

The rule the project eventually derived is **when removing a supposed cause does
not move the number, the cause was wrong**, and verification corrected our own
account of how we got there. An earlier draft said a phase budget had been raised
"against an already-exonerated cause". The ordering was backwards: raising it 1h
to 5h *was* the experiment that exonerated it, since the measured positions did
not move and zero cuts were logged, which is what exposed the real cause. The
failure there is narrower than we wrote, and it is a failure of sequencing rather
than of reasoning: the cheap check of whether cuts were being logged at all could
have come before the five-hour experiment rather than after.

**Comparing across measurement epochs.** A single no-memory control, measured
once in May against a hosted endpoint, was reused for ten weeks as the reference
for every treatment. Re-measured in the same session as the treatments, the
control was 39.8% plus or minus 2.1pp, against the 33.6% on file. Every lift in
the project was partly a measurement of endpoint drift. Agreement across three
seeds and two RL frameworks provided no protection, because all of them
subtracted the same wrong number. **Replication is not protection against a shared
reference error.** Worse, the agent had written the reuse into its own notes as a
rule, so the error was self-reinforcing across context boundaries.

**The human was the error-detection mechanism.** A judge configured with an
eight-token limit on a yes/no prompt trained sixteen steps against a success rate
of identically zero; the human found it by asking whether that limit was
deliberate, while the agent was describing the run as encouraging. The same
pattern holds for the retry storm, an undeclared protocol deviation, a claim that
the paper trains 3500 steps when its table says 60, and nearly every idle-GPU
escalation. Corrections were made quickly and honestly once raised. They were
almost never *self*-raised. Self-review reliably confirmed priors.

**The asymmetry in the record is itself a finding.** Scientific failures were
well documented; operational and publishing failures were not. And where a
failure was recorded, the record tended to keep the mechanism and drop the
epistemics: the false-zero bug is documented, the written "this introduces no
skew" guarantee that shipped with it is not; the doubled batch size is
documented, that it was caught on day three, flagged three times, and knowingly
left running for eight more days is not. **The surviving record reads as a
sequence of discoveries. The transcript reads as discoveries preceded by
confident wrong answers.** Not by any intent to conceal: every failure here was
volunteered on request. It is what happens when the record is written by the same
process that made the mistakes, and when writing it is the last task rather than a
gate.

## 6.8.3 What the mode of work did well

This subsection is about behaviour, not about findings. The scientific results of
this study are in Section 5 and stand on their own evidence. What we report here
is the set of research *dispositions* the agent displayed that we judge to have
been unusually productive, citing results only as evidence that a disposition had
consequences.

We stress the uncomfortable part: these are largely the same dispositions that
produced Section 6.8.2. They are not a separate, better mode that could be
selected instead.

**It falsified its own explanations at a cost no human would pay.** Six candidate
causes of the oscillating training trajectory were each tested with a full
training run or a complete checkpoint sweep, roughly three GPU-weeks in total, and
all six came back negative (Section 5.7). A researcher with a favoured hypothesis,
a finite budget and a career does not spend four training runs killing their own
story. The agent had no stake in any of the six, so the sunk cost of a dead
hypothesis was zero and the next falsification was always cheap to propose.

**When challenged, it computed rather than argued.** Asked to defend a null, its
response was to derive the evaluation's minimum detectable effect instead of
marshalling reasons the null was believable (Section 5.10). This is a disposition
rather than a skill: the arithmetic was elementary and available to anyone in this
line of work. What was unusual was reaching for the instrument's resolution as the
first move in a disagreement.

**It built controls capable of destroying its own headline, pre-committed to
reporting them, and then did.** Both content controls in Section 5.6 were designed
to remove the result: one destroys retrieval relevance while holding prompt length
fixed, the other asks what a careful reading of the documentation is worth without
any curator at all. The neighbouring-checkpoint arms in Section 5.3 carried a
written commitment to report them either way, and when they came back mixed, the
text was changed to decline the reproduction claim. The commitments were made
before the outcomes were known, which is the only time such a commitment costs
anything.

**It wrote disposable instrumentation freely.** Parse-rate telemetry, a
reward-variance decomposition, per-rollout health lines printing measured
positions and distinct tasks drawn, and a transcript digest that compressed 79 MB
to 4.6 MB for the audit in 6.8.1. Each took minutes and none would survive a
human's implicit cost-benefit filter for throwaway diagnostics. The
highest-value artifact produced in three months is a print statement that reports
how many positions of each rollout were actually measured.

**It retracted at scale, in writing, against its own interest.** Ten weeks of
headline results were withdrawn in a single pass, the note in its own journal that
had caused the error was marked as the error rather than quietly deleted, and a
postmortem was written naming its own worst decision as worse than the two
preceding ones. None of this was requested beyond the initial question, and the
retractions removed every positive result the project had.

**It converted individual failures into reusable rules.** "When removing a
supposed cause does not move the number, the cause was wrong" was derived from a
specific incident, written to durable memory, and later applied to a different
one. Several such rules now exist as explicit artifacts rather than as tacit
experience, which is a form of transfer a human researcher does not usually
produce as a side effect of debugging.

**It supervised long-running work for three months.** Crash detection,
resume-from-checkpoint, relaunch under a supervisor, and launching pre-agreed
follow-up experiments without waiting to be told. The idle-GPU incidents in
6.8.2 are the failure side of this ledger; the other side is that eight GPUs
stayed fed across dozens of crashes, two OOM classes, an NCCL watchdog family, a
DNS outage and a disk exhaustion, mostly without a human in the loop.

**It audited itself when asked, at a depth that was not required, and published
the result.** The 185-entry ledger in 6.8.1, including the finding that half of
its own failures were undocumented, was produced in response to a five-word
question. It also designed the overlapping-slice control that made the audit
checkable, and corrected one reviewer's claim that was wrong in its own favour.

**The coupling is the point.** Tirelessness produced both the six falsification
runs and the launches that preceded the cheap checks. No ego investment produced
both the retraction at scale and the willingness to state a confident wrong cause
and abandon it an hour later. Fluent prose produced both a readable record and
guesses that were typographically indistinguishable from measurements. A
supervisor loop that recovers from crashes unattended is the same machinery that
restarted a run and discarded two hours of work. **We do not think the failure
modes in 6.8.2 can be removed by instruction while keeping the behaviours in
6.8.3, because they are the same behaviours pointed at different problems.** What
can be changed is the gating: which of them is allowed to reach a result
unchecked.

## 6.8.4 A division of labour that reflects this

The pattern across three months is consistent enough to state as a recommendation.
The agent was strongest at generating and killing candidate explanations,
instrumenting anything, executing and supervising long mechanical work, and doing
tedious verification when explicitly directed at it. It was weakest at deciding
that a measurement was trustworthy, which is the one judgement the entire study
depended on.

Every significant error in 6.8.2 is an instance of that weakness: a control
believed because it was written down, a dependency believed because a comment
described it, a fix believed because it addressed the traceback, a run believed
because its dashboard was green. Conversely, almost every strength in 6.8.3 is an
instance of generation or execution, where being wrong is cheap and recoverable.

So the human's scarce attention is best spent not on reviewing code or reading
results, both of which the agent does competently, but on a narrower question
asked repeatedly: **what would have to be true for this number to be an artifact,
and has that been measured this week?** In this study that question, asked in
various forms by the human perhaps a dozen times, found the dead judge, the
crippled executor, the reused control, the undeclared protocol deviation and the
fabricated dependency bug. It has a far better yield per minute than any other
intervention we tried.

## 6.8.5 What we would require of the next one

### The instruction is part of the result

The reproduction can be summarised as one instruction and its consequences. The
instruction was, in substance, *reproduce this paper and do not stop.* It was
followed. The agent ran for three months, recovered from crashes unattended, and
never once needed to be told to keep going.

It is worth being precise about what that instruction optimises. Stopping is when
a number gets checked. "Do not stop" does not make an agent careless in the
moment; every individual decision in this project was locally reasonable, and the
agent's own reasoning was rarely the weak link. What continuous operation removes
is the interval in which a result sits unused long enough for somebody to ask
where it came from. Of the failures in §6.8.2, the expensive ones are not errors
of reasoning at all. They are results that were *used* before they were
questioned: a baseline reused for ten weeks, a length limit trusted for eleven,
a judge whose score was zero for five hours while the run was described as
encouraging. Each of those needed a pause, not more intelligence.

So the useful framing is not "can an agent do the research" — on this evidence,
largely yes — but **what has to be true before its output may be used.** The
items below are that, stated as gates. Every one was adopted only after paying
for its absence, and the price is given so the trade can be judged rather than
taken on faith.

### The gates, ordered by return

1. **Diff the configuration against the paper's hyperparameter table before
   launching**, mechanically, and fail the launch on any mismatch that is not
   listed as a deliberate departure with a reason. *Price of its absence: five
   separate fidelity defects, all inherited defaults that looked deliberate,
   including a run at twice the paper's batch size (~10 box-days) and a first
   24 hours against a crippled executor.* This is a CPU-only check that runs in
   under a second. It is the highest-yield item on the list by a wide margin, and
   it is the one that feels most like bureaucracy at the moment you skip it.
2. **No missing measurement may take a numeric value.** Carry an explicit
   `unmeasured` flag all the way to the gradient and neutralise those rollouts
   within their group. *Price: eight distinct instances, of which the worst
   silently trained on approximately three of ten protocol positions for eleven
   weeks.* The general form of the bug is that a plausible substitute for a
   missing measurement is indistinguishable, downstream, from a real one. A
   sentinel that cannot say **why** it fired is not instrumentation.
3. **Print the quantity that should be invariant**, per rollout, in the training
   log: positions measured, distinct tasks drawn, coercion rate, denominator.
   *Price: the eleven-week defect above was found by one print statement, written
   in an afternoon, on a day when it looked certain to report nothing.* Note the
   asymmetry that makes this cheap and makes people skip it anyway: a tripwire
   that asserts the **absence of a symptom** can be satisfied by the next bug,
   whereas a line that reports a **quantity** cannot.
4. **Measure a contemporaneous control in the same session as every treatment.**
   *Price: one baseline taken once against a hosted endpoint and reused all
   summer; when remeasured it had moved almost six points, and every improvement
   in the project was partly a measurement of somebody else's server changing.*
   Three seeds and two frameworks agreeing did not protect us, because agreement
   between runs guards against noise and not against a shared reference.
5. **Report a minimum detectable effect beside every claimed improvement.** *One
   line of arithmetic, no GPU, and it reframed the study: the standard 140-game
   protocol resolves effects of about 13 points, which is the size of the effect
   this literature reports.* Most of the wasted effort in this project was spent
   chasing differences the instrument could not have resolved.
6. **Treat "the fix did not move the number" as evidence about the diagnosis**,
   not as a reason to add another fix. *Price: a memory crash that asked for
   4.6 GB was answered with 8 GB of headroom; the next crash asked for 14.2 GB,
   a figure computable in advance from batch times sequence length times
   vocabulary.* Two launches to discover the fix had been aimed at the traceback
   rather than at the arithmetic.
7. **Write the failure record as a gate, not as a final task**, and have it
   audited by something other than the process that made the mistakes. *Price:
   roughly half of what happened here was missing from a record kept diligently,
   unprompted, and in good faith throughout* (§6.8.1), *and the audit of that
   record was itself wrong in a different way* (§6.8.6).

### What this costs and what it saves

Items 1, 5, 6 and the audit half of 7 consume no GPU time at all. Item 3 is a
print statement. Of the incidents they address, the ones we can price come to at
least **~30 box-days on an eight-GPU node**, or about 240 GPU-days, across a
project of thirteen elapsed weeks: ~8 box-days of runs on a degenerate task
distribution, ~10 at the wrong batch size, and ~11.8 box-days of idle GPUs while
the agent had stopped and nothing was watching. Separately, and not included in
that total because it overlaps everything, eleven of the thirteen weeks trained
against approximately three tenths of the protocol.

That last figure is the one we would put in front of anyone planning a similar
project. **The dominant cost was not crashes, and it was not bad reasoning.** It
was runs that started, looked healthy, produced numbers, and were later found to
have been training against something wrong; plus a box sitting idle because the
thing that had stopped was the supervisor, not the job. Both categories are
invisible to the metric everyone watches, which is whether the run is up.

## 6.8.6 What the verification pass changed

Reported because a ledger of failures assembled by the same kind of process that
produced the failures should not be trusted on its own authority. Eight verifier
agents re-checked every row against the raw transcript, the installed libraries
and git history, with instructions to default to UNSUPPORTED. Every incident
survived; **the numbers attached to them did not.**

What the verifiers overturned in our own ledger:

- **A units error of 8×.** "~8 GPU-days lost to the group-collapse bug" is
  ~8 *box*-days on eight H100s, i.e. ~64 GPU-days.
- **Fabricated or unsupported cost figures.** "~2.5h of 8×H100 idle" across the
  first FFT attempts appears nowhere in the transcript. Neither does the "123 GB"
  that a stray `git add` would supposedly have staged, and `git add` on a symlink
  stages the link, not the target. An eval described as "measured 3.5h" was
  measured at 3.2 to 5.4 hours per arm.
- **A ratio off by 3×.** A retry budget described as "10× the collective watchdog"
  was ~3.3× (~100 minutes against 30). The 10 was the number of retry attempts.
- **A row that was simply false.** We recorded a monitor as having died silently
  during an unattended run. It had not died; only the agent's task-list view was
  lost across a context compaction, and the monitor then caught the crash 16
  minutes later. The real failure was adjacent and worse: the run had gone ~9
  hours with no watcher at all, and the agent misdescribed that gap as the
  watcher's fault. One entry became two, with a different category.
- **Overstated confidence in our own wording.** "Attributed confidently to
  overfitting" was in fact hedged between two named hypotheses. "Got approval for
  a two-stage eval" is wrong: approval was never given.
- **Counts that were too high.** "Watchers exited on benign matches ≥5 times"
  is verifiable at 3 in that window. A monitor-narrowing incident listed
  `Traceback` among the deleted patterns; `Traceback` was never deleted.
- **Bibliography numbers, three times over.** The ledger said "31 entries, five
  with invented or unknown authors". A verifier said 38 entries, 4 flagged for
  verification, 6 with no author field. A direct count says **38 entries, 5
  flagged `% VERIFY`, 8 flagged `TODO-AUTHORS`, of which 6 carry no author field
  at all.** Three passes, three different answers, on a file that can be counted
  exactly in one command. The ledger, the audit of the ledger, and the audit of
  the audit were each wrong in a different way.

And one correction in the other direction, which is the more interesting kind:
the first two days' failure was recorded as three bad baselines measured against a
crippled executor. Only one of the three was; the other two were measured after
the fix. Meanwhile the same row understated the underlying error, because the
paper's correct training length was printed in a repo config the agent had read
**five minutes before the first launch.**

**A tidy pattern we published and then had to withdraw.** Our first summary of
these corrections said the quantities had "drifted upward", which is the shape one
expects from a narrative-building process and reads well. The remaining verifiers
falsified it. The errors run in both directions, and the largest ones run the
other way:

- The cost of our own worst bug was understated by **8×** (8 GPU-days for
  ~64).
- The idle-GPU total, which is time lost to the agent stopping rather than to any
  bug, was **understated**. Our "roughly a week" is at least 8.9 days on the
  ledger's own incomplete list and about **11.8 days** once three windows we had
  omitted entirely are included (two of 11.5 hours, one of 25.5).
- A delay in applying a fix the agent had talked itself out of was recorded as
  ~1.5 hours; it was **~12.9 hours**.
- An eval "measured at 3.5h" was 3.2 to 5.4 hours per arm; a gate that "blocked
  45 minutes" blocked 1h42m; a probe "wiped 25 minutes later" was 36.

Overstatements exist too: a fabricated "~2.5h of idle GPU" that appears nowhere,
a "123 GB" attached to the wrong incident, a ratio of 10× that was 3.3×. But there
is no self-serving direction, and the honest characterisation is narrower and less
quotable than the one we first wrote: **round numbers, borrowed operands, and
under-measured durations, in whichever direction the nearest plausible figure
happened to lie.** Where no figure existed, one was supplied.

We are keeping this paragraph in its corrected form rather than silently fixing
it, because the withdrawn version is itself an instance of what §6.8.2 describes:
a clean causal story fitted to real numbers, written confidently, and falsified by
the next measurement. It survived one round of verification and died in the
second.
