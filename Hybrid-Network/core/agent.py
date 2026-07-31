"""
agent.py — Type-L (LLM-driven) agent.

One LLM call per agent per step. That call produces the agent's new opinion
text, its reasoning, and its own updated memory. There is no separate
summarisation call and no word-window truncation: the agent consolidates its
memory itself, so information from every round can reach later rounds.

The agent does not emit a numeric position. `belief` is filled in by the model
after each step by scoring `opinion` with the stance regressor (core/scorer.py),
and is a continuous value in [-1, +1] on the same scale as Type-C agents.

Messages are split so the cacheable part comes first: the system message holds
the task instructions and field definitions and is byte-identical for every
agent, while persona and this round's content go in the user message. Prefill
dominates the step (50 agents x ~930 prompt tokens against ~290 decode tokens
each), so letting vLLM reuse the shared prefix removes real work rather than
just tidying the prompt.
"""

from mesa import Agent

from prompt import (update_opinion_prompt, system_message, PERSONA_TEMPLATE,
                    LEADER_CLAUSE)
from utils import get_completion_from_messages_structured, SCHEMA_BY_ORDER


def _truncate_words(text, max_words):
    words = (text or "").split()
    return " ".join(words[:max_words]) if len(words) > max_words else (text or "")


class SocialAgent(Agent):
    def __init__(self, model, unique_id, name, gender, age, traits, qualification,
                 initial_belief, topic, gpt_model, temp=0.0, initial_opinion=None,
                 initial_reasoning=None, is_leader=False,
                 field_order="cot_first", heard_word_cap=60, memory_word_cap=120):
        super().__init__(unique_id, model)
        self.unique_id = unique_id

        self.name = name
        self.gender = gender
        self.age = age
        self.traits = traits
        self.qualification = qualification
        self.topic = topic
        self.is_leader = is_leader

        self.gpt_model = gpt_model
        self.temp = temp
        self.field_order = field_order
        self.heard_word_cap = heard_word_cap
        self.memory_word_cap = memory_word_cap

        # Identical for every agent in the run, so vLLM can cache its KV.
        self.system_prompt = system_message(field_order)
        self.persona = PERSONA_TEMPLATE.format(
            name=name, age=age, gender=gender, traits=traits,
            qualification=qualification)

        # Fixed anchor — never overwritten.
        self.initial_opinion = initial_opinion
        self.initial_reasoning = initial_reasoning

        # belief is the canonical continuous position in [-1, 1]; the model fills
        # it in from the stance regressor after each step.
        self.belief = float(initial_belief)
        self.beliefs = [self.belief]
        self.opinions = [self.initial_opinion]
        self.reasonings = [initial_reasoning]

        self.memory = ""
        self.long_memory_full = [self.memory]

        self.heard = []
        self.short_memory_full = []

        self.agent_interaction = []
        self.contact_ids = []

        self.parse_failed = False

    # ── Phase 1: listen (no LLM call) ───────────────────────────────────────
    def interact(self):
        """Collect neighbours' latest opinions. Reads only pre-update state."""
        heard, contacts = [], []
        for other in self.agent_interaction:
            contacts.append(other.unique_id)
            heard.append(_truncate_words(other.opinions[-1], self.heard_word_cap))

        self.contact_ids.append(contacts)
        self.heard = heard
        self.short_memory_full.append(heard)
        self.agent_interaction = []

    def step(self):
        self.interact()

    # ── Phase 2 helper ──────────────────────────────────────────────────────
    def build_user_message(self):
        heard = "\n".join(f"- {op}" for op in self.heard) or "- (you heard nothing this round)"
        return update_opinion_prompt.format(
            persona=self.persona,
            leader_clause=LEADER_CLAUSE if self.is_leader else "",
            topic=self.topic,
            initial_opinion=self.initial_opinion or "",
            current_opinion=self.opinions[-1] or "",
            memory=self.memory or "(nothing yet)",
            heard=heard,
        )

    # Kept for diagnostics: the full prompt as the model sees it.
    def build_prompt(self):
        return self.system_prompt + "\n\n" + self.build_user_message()

    # ── Phase 2: update (one LLM call) ──────────────────────────────────────
    def update(self):
        """Single LLM call: new opinion, reasoning, and updated memory.

        On failure the agent keeps its previous opinion and memory. That is
        indistinguishable from a deliberate decision to hold, so `parse_failed`
        is recorded and reported — an elevated failure rate inflates apparent
        stubbornness and must be controlled for when comparing models.
        """
        response = get_completion_from_messages_structured(
            messages=self.build_user_message(),
            system_messages=self.system_prompt,
            model=self.gpt_model,
            temperature=self.temp,
            response_type=SCHEMA_BY_ORDER[self.field_order],
        )

        if response is None:
            self.parse_failed = True
            self.opinions.append(self.opinions[-1])
            self.reasonings.append("")
            self.long_memory_full.append(self.memory)
            return

        self.parse_failed = False
        self.memory = _truncate_words(response.memory, self.memory_word_cap)
        self.opinions.append(response.opinion)
        self.reasonings.append(response.reasoning)
        self.long_memory_full.append(self.memory)

    # ── Called by the model after stance scoring ────────────────────────────
    def set_belief(self, stance):
        self.belief = float(stance)
        self.beliefs.append(self.belief)
