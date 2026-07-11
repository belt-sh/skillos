"""CoT executor prompt for the reasoning benchmark.

Paper Appendix A.2 / Fig 11 (verbatim not in the reconstructed .md — kept short
and standard-CoT here; deviations vs Fig 11 are a source of variance to note
in the report but not a bug). Skills are injected as retrieved context ahead
of the problem, same slot as the ALFWorld executor prompt.
"""

REASONING_SYSTEM_AIME = (
    "You are an expert mathematician. Solve the problem step by step. "
    "Show your reasoning clearly. The final answer is an integer between "
    "000 and 999. Put your final answer inside \\boxed{...}."
)

REASONING_SYSTEM_GPQA = (
    "You are a scientific expert. Read the multiple-choice question and "
    "reason step by step. Choose exactly one letter A, B, C, or D. Put your "
    "final answer letter inside \\boxed{...}."
)

REASONING_USER_TEMPLATE = (
    "Relevant skills from the repository:\n"
    "{past_skills}\n\n"
    "Problem:\n"
    "{problem}\n\n"
    "Reason step by step, then give your final answer in \\boxed{{...}}."
)


REASONING_SYSTEM_GPQA_FT = (
    "You are a scientific expert. Read the question and reason step by step. "
    "Give the exact answer (e.g. a value with units, a compound name, or a "
    "short phrase). Put your final answer inside \\boxed{...}."
)


def build_messages(problem: str, past_skills: str, kind: str) -> list[dict]:
    system = (
        REASONING_SYSTEM_AIME if kind == "aime"
        else REASONING_SYSTEM_GPQA_FT if kind == "gpqa_ft"
        else REASONING_SYSTEM_GPQA
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": REASONING_USER_TEMPLATE.format(
            past_skills=past_skills or "(no skills yet)",
            problem=problem)},
    ]
