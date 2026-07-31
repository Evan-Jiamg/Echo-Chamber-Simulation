#!/usr/bin/env python3
"""
reddit_init.py  —  Bootstrap N=50 agents from Reddit stance data.

Stratified-samples 50 agents per topic from the RoBERTa-scored Reddit
parquet files, then calls the local Phi-4 (vLLM) to generate initial
opinion text and agent personas.

Outputs (written to data/):
  numeric_sim_opnions_and_stubbornness_num_agents_50_{topic}.json
  agents_backgrounds_num_agents_50_{topic}_phi4.json

Usage:
  python3 reddit_init.py --topic gun_control
  python3 reddit_init.py --topic abortion
  python3 reddit_init.py --topic gun_control --dry_run   # skip LLM calls
"""

import argparse
import json
import os
import random
import sys

import numpy as np
import pandas as pd
from openai import OpenAI
from names_dataset import NameDataset

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE      = os.path.dirname(os.path.abspath(__file__))
_PROJ      = os.path.normpath(os.path.join(_HERE, os.pardir, os.pardir))
REDDIT_DIR = os.environ.get("REDDIT_DATASET",
                            os.path.join(_PROJ, "Reddit-Dataset"))
# data/agents/, not data/: the priors moved when the tree was regrouped
SIM_DATA   = os.path.join(_PROJ, "Hybrid-Network", "data", "agents")

VLLM_URL   = "http://localhost:11434/v1"
PHI4_MODEL = "phi4"

# ── Topic config ──────────────────────────────────────────────────────────────
# Clusters are pre-identified from HDBSCAN output on r/politics
TOPIC_CONFIG = {
    "gun_control": {
        "clusters": [775, 708, 750, 725, 705, 456],
        "label":    "gun control",
        "question": "Should the government implement stricter gun control laws?",
    },
    "abortion": {
        "clusters": [132, 131],
        "label":    "abortion",
        "question": "Should abortion be legal and accessible?",
    },
}

# ── Stratification ─────────────────────────────────────────────────────────────
# Thresholds that split continuous stance ∈ [-1,+1] into 5 ordinal bins
BIN_THRESHOLDS = [-0.75, -0.25, 0.25, 0.75]  # 5 bins: [-1,t0), [t0,t1), [t1,t2), [t2,t3), [t3,1]
BIN_LABELS     = [-2, -1, 0, 1, 2]

BELIEF_KEYWORDS = {
    -2: ["Firmly reject", "Strongly disagree", "Completely oppose"],
    -1: ["Mildly reject", "Somewhat disagree", "Gently oppose"],
     0: ["Objectively consider", "Impartially acknowledge", "Undecidedly weigh"],
     1: ["Somewhat agree", "Mildly support", "Generally favor"],
     2: ["Strongly agree", "Fully support", "Wholeheartedly favor"],
}

EDUCATION_LEVELS = [
    "High School", "Some College", "Bachelor's Degree",
    "Master's Degree", "Doctorate",
]

PERSONALITY_TRAITS = [
    "Openness", "Conscientiousness", "Extraversion", "Agreeableness",
    "Neuroticism", "Curiosity", "Empathy", "Resilience", "Creativity",
    "Discipline", "Naturalism", "Punctuality", "Playfulness", "Emotionality",
]

N_AGENTS = 50


# ── Stratified sampling ────────────────────────────────────────────────────────

def load_topic_stances(topic: str) -> pd.DataFrame:
    cfg = TOPIC_CONFIG[topic]
    frames = []
    for cid in cfg["clusters"]:
        fpath = os.path.join(REDDIT_DIR, "stance_scores",
                             f"cluster_{cid}.parquet")
        if os.path.exists(fpath):
            frames.append(pd.read_parquet(fpath))
        else:
            print(f"  [WARN] Missing parquet: cluster_{cid}.parquet")
    if not frames:
        raise FileNotFoundError(f"No parquet files found for topic '{topic}'")
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset="comment_id")
    print(f"  Pooled {len(df)} unique comments for topic='{topic}'")
    return df


def assign_bin(stance: float) -> int:
    for i, threshold in enumerate(BIN_THRESHOLDS):
        if stance < threshold:
            return BIN_LABELS[i]
    return BIN_LABELS[-1]


def stratified_sample(df: pd.DataFrame, n: int = N_AGENTS, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["bin"] = df["stance"].apply(assign_bin)

    total = len(df)
    counts = df["bin"].value_counts().sort_index()
    proportions = counts / total

    # Allocate agents per bin proportionally, enforce sum = n
    raw_alloc = {b: proportions.get(b, 0.0) * n for b in BIN_LABELS}
    alloc = {b: int(v) for b, v in raw_alloc.items()}
    deficit = n - sum(alloc.values())
    # Give remaining slots to bins with largest fractional remainder
    remainders = sorted(raw_alloc.keys(), key=lambda b: -(raw_alloc[b] - alloc[b]))
    for b in remainders[:deficit]:
        alloc[b] += 1

    print("  Allocation per bin:", alloc)

    sampled = []
    for b in BIN_LABELS:
        n_b = alloc[b]
        if n_b == 0:
            continue
        pool = df[df["bin"] == b]
        if len(pool) < n_b:
            print(f"  [WARN] bin={b}: only {len(pool)} comments, want {n_b} — sampling with replacement")
            rows = pool.sample(n=n_b, replace=True, random_state=int(rng.integers(1000)))
        else:
            rows = pool.sample(n=n_b, replace=False, random_state=int(rng.integers(1000)))
        sampled.append(rows)

    result = pd.concat(sampled, ignore_index=True)
    assert len(result) == n, f"Expected {n} agents, got {len(result)}"
    return result.reset_index(drop=True)


# ── Persona generation ─────────────────────────────────────────────────────────

def make_persona(agent_id: int, nd: NameDataset) -> dict:
    gender = random.choice(["male", "female"])
    gender_code = "M" if gender == "male" else "F"
    try:
        names = nd.get_top_names(n=20, gender=gender_code, country_alpha2="US")
        first = random.choice(names["US"][gender_code])
    except Exception:
        first = f"Agent_{agent_id}"
    name = f"{first}_{agent_id}"
    age = random.randint(18, 65)
    education = random.choice(EDUCATION_LEVELS)
    traits = random.sample(PERSONALITY_TRAITS, 5)

    system_prompt = (
        f"\n    Imagine you are a human. Your name is {name}, and your gender is {gender}. \n"
        f"    You are {age} years old. Your personality is shaped by these specific traits: {traits}\n"
        f"    Your education level is {education}.\n"
        f"    You are participating in an online discussion. "
        f"You must stay in character and express your views naturally."
    )
    return {
        "name": name,
        "age": age,
        "education level": education,
        "traits": [", ".join(traits)],
        "gender": gender,
        "system_prompt": system_prompt,
    }


# ── LLM opinion generation ────────────────────────────────────────────────────

def build_opinion_prompt(topic_label: str, belief_int: int) -> str:
    keyword = random.choice(BELIEF_KEYWORDS[belief_int])
    return (
        f"Given the topic '{topic_label}', your belief value is {belief_int}.\n"
        f"Provide your opinion ensuring you include the keyword '{keyword}'.\n"
        f"Begin with 'I {keyword}...' and write 1–2 sentences (max 120 words).\n"
        f"Output ONLY the opinion text, nothing else."
    ), keyword


def generate_opinion_and_reasoning(
    client: OpenAI,
    topic_label: str,
    belief_int: int,
    persona: dict,
    temperature: float = 0.5,
) -> tuple[str, str]:
    opinion_prompt, keyword = build_opinion_prompt(topic_label, belief_int)

    opinion_resp = client.chat.completions.create(
        model=PHI4_MODEL,
        messages=[
            {"role": "system", "content": persona["system_prompt"]},
            {"role": "user",   "content": opinion_prompt},
        ],
        max_tokens=160,
        temperature=temperature,
    )
    opinion_text = opinion_resp.choices[0].message.content.strip()
    # Ensure it starts with the keyword
    if not opinion_text.startswith("I "):
        opinion_text = f"I {keyword} {opinion_text}"

    reasoning_resp = client.chat.completions.create(
        model=PHI4_MODEL,
        messages=[
            {"role": "system", "content": persona["system_prompt"]},
            {"role": "user",   "content": (
                f"In 1–2 sentences (max 80 words), explain why you hold this position "
                f"on '{topic_label}': \"{opinion_text}\"\n"
                f"Output ONLY the reasoning, nothing else."
            )},
        ],
        max_tokens=120,
        temperature=temperature,
    )
    reasoning_text = reasoning_resp.choices[0].message.content.strip()
    return opinion_text, reasoning_text


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", required=True, choices=list(TOPIC_CONFIG.keys()),
                        help="Topic to initialise: gun_control | abortion")
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--temp",     type=float, default=0.5)
    parser.add_argument("--dry_run",  action="store_true",
                        help="Skip LLM calls; use placeholder text")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    cfg = TOPIC_CONFIG[args.topic]
    print(f"\n{'='*60}")
    print(f"  Topic : {args.topic}  ({cfg['label']})")
    print(f"  Seed  : {args.seed}")
    print(f"  DryRun: {args.dry_run}")
    print(f"{'='*60}\n")

    # ── Step 1: Load and sample ──────────────────────────────────────────────
    print("Step 1: Loading Reddit stance data...")
    df = load_topic_stances(args.topic)
    sampled = stratified_sample(df, n=N_AGENTS, seed=args.seed)

    print("\nSampled stance distribution:")
    print(sampled["bin"].value_counts().sort_index().to_string())
    print(f"Stance mean={sampled['stance'].mean():.3f}, std={sampled['stance'].std():.3f}\n")

    # ── Step 2: Build opinions/stubbornness JSON ─────────────────────────────
    print("Step 2: Assigning beliefs and stubbornness...")
    opinions = {}
    stubbornness = {}
    for i, row in sampled.iterrows():
        opinions[str(i)]     = float(np.clip(row["stance"], -1.0, 1.0))
        stubbornness[str(i)] = float(np.random.uniform(0.3, 0.7))

    op_path = os.path.join(
        SIM_DATA,
        f"numeric_sim_opnions_and_stubbornness_num_agents_{N_AGENTS}_{args.topic}.json"
    )
    with open(op_path, "w") as f:
        json.dump({"opinions": opinions, "stubbornness": stubbornness}, f, indent=2)
    print(f"  Saved: {op_path}")

    # ── Step 3: Generate agent backgrounds ──────────────────────────────────
    print(f"\nStep 3: Generating agent backgrounds ({'DRY RUN' if args.dry_run else 'Phi-4'})...")
    nd = NameDataset()
    client = None if args.dry_run else OpenAI(base_url=VLLM_URL, api_key="ollama")

    backgrounds = {}
    for i, row in sampled.iterrows():
        belief_int = int(row["bin"])
        persona    = make_persona(i, nd)

        if args.dry_run:
            kw = BELIEF_KEYWORDS[belief_int][0]
            opinion_text   = f"I {kw} the idea of {cfg['label']}. [placeholder]"
            reasoning_text = f"[placeholder reasoning for belief={belief_int}]"
        else:
            try:
                opinion_text, reasoning_text = generate_opinion_and_reasoning(
                    client, cfg["label"], belief_int, persona, args.temp
                )
            except Exception as e:
                print(f"  [WARN] Agent {i} LLM call failed: {e}")
                kw = BELIEF_KEYWORDS[belief_int][0]
                opinion_text   = f"I {kw} {cfg['label']}."
                reasoning_text = "No reasoning available."

        persona["initial_opinion"]   = opinion_text
        persona["initial_reasoning"] = reasoning_text
        backgrounds[str(i)] = persona

        stance_str = f"{row['stance']:+.3f}"
        print(f"  [{i:02d}] bin={belief_int:+d}  stance={stance_str}  → \"{opinion_text[:60]}...\"")

    bg_path = os.path.join(
        SIM_DATA,
        f"agents_backgrounds_num_agents_{N_AGENTS}_{args.topic}_phi4.json"
    )
    with open(bg_path, "w") as f:
        json.dump({"backgrounds": backgrounds}, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {bg_path}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Done. Run the simulation with:")
    print(f"  python3 run_hybrid.py \\")
    print(f"    --topic {args.topic} \\")
    print(f"    --backgrounds_label phi4 \\")
    print(f"    --gpt_model {PHI4_MODEL}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
