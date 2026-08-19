# 2. Background and related work

## 2.1 Externalised memory for frozen agents

A large body of work gives an agent a place to put what it learns, so that the
agent improves without gradient updates. Reflexion \cite{shinn2023reflexion}
writes verbal self-critiques into the next attempt's context. Voyager
\cite{wang2023voyager} accumulates an executable skill library in Minecraft.
ExpeL \cite{zhao2024expel} distils cross-task insights from prior trials.
Generative Agents \cite{park2023generative} maintain a reflective memory stream,
and MemGPT \cite{packer2023memgpt} treats context as a paged resource. Agent
Workflow Memory \cite{wang2024awm} induces reusable workflows from experience.

By 2026 this had become a dense area, with skill libraries specifically framed as
procedural memory and multiple systems for self-evolving them
\cite{skillfoundry2026,autoskill2026,trace2skill2026,rawexperience2026} and at
least one survey of the design space \cite{memorysurvey2026}. A systematic study
of model-generated agent skills \cite{rawexperience2026} is the closest in spirit
to what we do here.

Two properties of this literature matter for our purposes. First, the memory is
usually produced by prompting rather than trained, so its quality is bounded by
the model writing it. Second, evaluation is usually a single number on a small
fixed benchmark, with no confidence interval and no correction for the number of
configurations tried. Section 5.10 argues that the second property is a serious
problem at the effect sizes involved.

## 2.2 Training the memory writer

SkillOS \cite{ouyang2026skillos} is a direct response to the first property. The
component that writes skills is a separate policy trained with GRPO
\cite{shao2024deepseekmath}, rewarded by the downstream success of a frozen
executor that consumes the skills. This is an appealing design: it decouples the
model that learns from the model that acts, so the acting model can be frozen,
closed, or arbitrarily large.

The reported results are strong. On ALFWorld \cite{shridhar2021alfworld} a
trained 8B curator improves a frozen executor by 13.3 percentage points, reaches
61.2% absolute with a 32B executor, and is reported to outperform a
frontier-model curator at the same job. The paper also reports cross-domain
transfer, with a curator trained on mathematics improving ALFWorld performance.

If those results hold, the technique is unusually practical. That is why we tried
to reproduce it.

## 2.3 Reinforcement learning for LLM agents

GRPO \cite{shao2024deepseekmath} removed the value network from PPO-style RLHF
and became the default for reasoning training \cite{deepseekr1}. Applying it to
multi-step agents raises credit assignment problems that episode-level advantages
handle poorly; GiGPO \cite{feng2025gigpo} addresses this with a two-level
grouping scheme and reports large gains over GRPO on ALFWorld and WebShop
\cite{yao2022webshop}. We use both a GRPO implementation (TRL
\cite{vonwerra2020trl} with ZeRO-3 \cite{rajbhandari2020zero}) and a GiGPO
implementation (verl \cite{sheng2024hybridflow}), partly to bound framework
effects on our conclusions.

## 2.4 Reproducibility and statistical practice

Our findings are as much about measurement as about skill repositories, so we
lean on an older literature.

Deep RL results were shown to be highly sensitive to seeds, implementation
details, and reporting choices \cite{henderson2018matters}, and to require more
seeds than are typically run \cite{colas2018seeds}. Agarwal et al.
\cite{agarwal2021precipice} showed that point estimates on small benchmark suites
routinely support conclusions the data cannot bear, and proposed interval-based
reporting. Bouthillier et al. \cite{bouthillier2021variance} quantified how much
benchmark variance comes from sources other than the treatment.

In NLP specifically, Card et al. \cite{card2020power} performed the analysis that
most directly anticipates our Section 5.10: they computed statistical power for
common benchmarks and found that most attempted comparisons to state of the art
are underpowered, with typical test sets unable to resolve the differences being
claimed. Dror et al. \cite{dror2018hitchhiker} catalogue the significance-testing
practices this requires. We apply McNemar's test \cite{mcnemar1947} with Holm
\cite{holm1979} and Benjamini-Hochberg \cite{benjamini1995} corrections, and
report bootstrap intervals throughout, following this line of argument rather
than inventing anything.

One thread is newer and specific to the present moment. Models served over
hosted APIs are not stationary instruments; Chen et al. \cite{chen2023drift}
documented measurable behavioural change in a commercial model over a few months.
Agent evaluations now routinely place a hosted model inside the measurement loop,
often as the frozen component whose behaviour is assumed constant. Section 5.1
reports what this cost us: a control measured against a hosted executor moved
5.7 percentage points in ten weeks, which is larger than most effects claimed in
Section 2.1's literature, and reusing it silently converted endpoint drift into
apparent treatment effect across seven training runs.

To our knowledge no prior work quantifies hosted-endpoint drift as a
confound *inside an agent evaluation*, and we suggest that contemporaneous
control measurement should be a stated requirement for any benchmark that calls
a hosted model.
