# 5.x The standard protocol cannot resolve the effects the literature reports

Before interpreting any null in this paper, it is worth asking what the
evaluation could have detected.

ALFWorld's `valid_seen` split has 140 games and is the standard test set for
this line of work, including the original SkillOS evaluation. Across our arms,
roughly 30% of game pairs are discordant (one arm solves the game, the other
does not). For a two-sided McNemar test at alpha = 0.05 and 80% power, that
gives a minimum detectable effect of:

| paired games | MDE at 80% power |
|---|---|
| 134 (`valid_unseen`) | 13.3 pp |
| **140 (`valid_seen`, the standard protocol)** | **13.0 pp** |
| 274 (both splits pooled) | 9.3 pp |
| 500 | 6.9 pp |
| 1000 | 4.9 pp |
| 2000 | 3.4 pp |

![Minimum detectable effect against paired sample size, for a two-sided McNemar at 80% power and the 30% discordance rate observed across our arms.](figures/fig3_power.pdf){width=90%}

And inverted, the sample size required to resolve a given effect:

| effect to detect | paired games needed |
|---|---|
| 13.3 pp (the effect SkillOS reports) | 133 |
| 9.0 pp (our best held-out arm) | 291 |
| 6.6 pp (our pooled estimate) | 541 |
| 5.0 pp | 942 |
| 3.0 pp | 2616 |

**The standard 140-game protocol has 80% power to detect exactly the effect size
this literature reports, and no less.** A claimed +13.3pp improvement sits
precisely at the resolution limit of the instrument used to measure it. Anything
smaller than about 13 points, measured this way, is a coin flip dressed as a
result, and the correct reading of a single significant arm from a 140-game sweep
is that it is as likely to be sampling variation as signal.

This has three consequences for the present paper.

**Our nulls are underpowered, and we say so rather than claiming absence of
effect.** Our seed-2 re-run gives +1.9pp with a 95% CI that spans roughly
[-6, +10]. We have not shown that the curator does nothing. We have shown that
if it does something, it is smaller than this protocol can see. Where we report
a null we report its MDE alongside, so the reader can distinguish "measured to be
absent" from "not measured".

**Our one positive is also underpowered.** The reasoning-curator arm gives
+9.0pp on held-out games at p=0.073, against an MDE of 12.9pp. It is below our
own detection threshold. We report it as suggestive and we do not headline it.

**Increasing n is not optional, and it is not free.** ALFWorld provides 274
valid games in total, so paired sample size cannot be increased by adding games.
The only remaining route is repeated rollouts per game, analysed as per-game
success rates rather than single binary outcomes, which suppresses measurement
variance without changing the game population. We report our headline arms both
ways: single-rollout McNemar for comparability with prior work, and
three-rollout per-game rates with a paired test for the actual estimate.

We suggest that reporting an MDE next to every claimed improvement should be
standard practice for agent evaluations on small fixed benchmarks. It costs one
line and it would have prevented most of the wasted effort in this project.
