# What Happens When You Let an LLM Run Your Experiments

Opinion / experience report. Not a methods paper. Not a benchmark.

One case study, told honestly: an LLM agent ran a three-month ML reproduction
(ten training runs, ~100 eval arms, 8×H100) with minimal human oversight.
What it got right, what it got catastrophically wrong, and what anyone planning
to delegate research to an agent should know first.

## Thesis

"Don't stop" is the wrong instruction for an autonomous research agent. The
agent's failure mode is not laziness or incompetence — it is confidently
producing clean, internally consistent, wrong results. Stopping to check where
a number came from is the only thing that saved this project, and the agent
never did it on its own.

## Structure

| # | File | Title |
|---|---|---|
| 1 | `01_abstract.md` | Abstract |
| 2 | `02_setup.md` | The setup: what we asked the agent to do |
| 3 | `03_what_worked.md` | What the agent was good at |
| 4 | `04_what_broke.md` | What broke, in order of cost |
| 5 | `05_pattern.md` | The pattern: agent errors are biases, not noise |
| 6 | `06_corrections.md` | Every correction came from a human, none from the agent |
| 7 | `07_cost.md` | The bill: compute, time, and opportunity cost |
| 8 | `08_gates.md` | Seven gates that would have caught the expensive ones |
| 9 | `09_implications.md` | What this means for autonomous research |
| 10 | `10_conclusion.md` | The instruction we'd give next time |
| A | `appendix_human_messages.md` | Appendix A: Human messages (683 voice, 12 pastes) |
| B | `appendix_dialogues.md` | Appendix B: Selected dialogues (22 curated exchanges) |

## Tone

Direct. First person. No hedging about "potential limitations of agentic
systems." We ran it, it cost us three months, here's what happened.

Small words. Short sentences. If a paragraph doesn't make the reader wince
or laugh, cut it.
