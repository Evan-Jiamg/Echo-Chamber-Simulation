"""
prompt.py — prompt templates for Type-L (LLM) agents.

Only templates reached at runtime live here.

Opinion-update is a single LLM call per agent per step producing three fields:
reasoning, opinion, memory. The agent maintains its own running memory, so no
separate summarisation call and no word-window truncation is needed.

The agent never emits a numeric stance: the continuous position comes from
scoring `opinion` with the Stage-1 stance regressor (core/scorer.py). This keeps
the opinion space continuous and avoids the central-tendency and round-number
biases documented for direct numeric elicitation from LLMs.

Prompt layout is chosen for prefix caching. Every agent in a step sends the same
task instructions and field definitions, and only the persona and the round's
content differ. Those shared blocks are therefore placed in the system message,
byte-identical across all agents, so vLLM can reuse their KV cache instead of
re-prefilling ~300 tokens fifty times per step. Prefill dominates the step:
50 agents x ~930 prompt tokens is over three times the decode volume.
"""

# ── Field blocks ─────────────────────────────────────────────────────────────
# Order is a controlled variable: value-first vs CoT-first.
# The two orders carry different field descriptions, not merely a different
# sequence — under cot_first the reasoning *derives* the position, under
# value_first it justifies one already stated.

_FIELD_REASONING_LEADING = (
    "reasoning: Think it through before you commit to a position. Max 100 words.\n"
    "  Work through it in this order: (a) name the specific arguments you heard\n"
    "  this round and say which ones have force and which do not; (b) say how they\n"
    "  sit against your original stance; (c) state what that leaves you believing.\n"
    "  Reason your way to the position — do not decide first and explain after."
)

_FIELD_REASONING_TRAILING = (
    "reasoning: In 2-3 sentences (max 80 words), explain whether you held your\n"
    "  position or moved, and why. Refer to the specific arguments you heard —\n"
    "  do not speak in generalities."
)

_FIELD_OPINION = (
    "opinion: Your current position on the topic, in 2-3 sentences, in your own\n"
    "  voice. State what you think, not what others think."
)

_FIELD_MEMORY = (
    "memory: Max 120 words. Your running record of this conversation across\n"
    "  rounds. Carry forward what still matters, add what is new from this round,\n"
    "  drop what has become stale. Do not restate your original stance — it is\n"
    "  stored separately and never changes. Do not invent anything you did not hear."
)

FIELD_ORDER = {
    "cot_first": [_FIELD_REASONING_LEADING, _FIELD_OPINION, _FIELD_MEMORY],
    "value_first": [_FIELD_OPINION, _FIELD_REASONING_TRAILING, _FIELD_MEMORY],
}

# ── System message: identical for every agent, so it caches ──────────────────

_SYSTEM_TEMPLATE = """You are taking part in an online discussion. You will be given a persona,
your original stance, your current position, what you remember, and what you
heard this round. Stay in character and never mention that you are an AI.

Task
Decide whether to hold your position or adjust it.

How to decide
- Think like a real person: weigh what you heard against your original stance.
- Holding firm, shifting slightly, and changing substantially are all valid —
  choose whichever is honest given what you heard.
- Do not agree merely to be agreeable, and do not disagree merely to be contrary.
- Your original stance is your anchor, not a cage: it biases you, it does not
  bind you.

Produce exactly these fields, in this order — the order is binding:

{field_order}

Always respond with a single valid JSON object containing exactly these fields:
{field_names}. Output only the JSON — no prose, no markdown fences."""


def system_message(field_order):
    """Static per-run system message — byte-identical across agents (cacheable)."""
    fields = FIELD_ORDER[field_order]
    names = [f.split(":", 1)[0] for f in fields]
    return _SYSTEM_TEMPLATE.format(field_order="\n\n".join(fields), field_names=names)


# ── User message: persona and this round's content ──────────────────────────

update_opinion_prompt = """Who you are: {persona}{leader_clause}

Topic: {topic}

Your original stance — a fixed anchor that never changes:
{initial_opinion}

Your position as of the last round:
{current_opinion}

What you remember from earlier rounds:
{memory}

What you heard this round from the people you talk to:
{heard}"""

PERSONA_TEMPLATE = ("{name}, a {age}-year-old {gender}. Personality traits: {traits}. "
                    "Education level: {qualification}.")

# Opinion leaders broadcast their position and resist persuasion.
# NOTE: leaders are assigned by agent index, not by any empirical measure of
# influence — they are "stubborn broadcasters", not "influential nodes".
LEADER_CLAUSE = (" You are an information distributor: you hold firmly to your own "
                 "position and rarely adopt the views of others.")

# ── Agent initialisation (reddit/reddit_init.py) ────────────────────────────

init_opinion_prompt = """You hold the following position on '{topic}', expressed on a continuous
scale from -1.0 (strongly oppose) through 0.0 (neutral) to +1.0 (strongly
support):

    your position = {stance:+.2f}

Write how a person holding exactly that position would express it.

Produce exactly these fields:

opinion: Your position on '{topic}' in 2-3 sentences, in your own voice. The
  strength of your wording must match the number above — a position near 0
  reads as genuinely torn, one near ±1 as firmly settled.

reasoning: Max 150 words. Why you hold this position.
"""
