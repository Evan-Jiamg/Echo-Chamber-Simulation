# EchoChamberSim

A multi-mode social opinion dynamics simulator combining LLM-driven agents (GPT-4o-mini) with numeric update rules (DeGroot, Friedkin-Johnsen, BCM) on dynamic K-NN networks. Built to study echo chambers, polarization, and the Price of Anarchy (PoA) in social networks.

> Based on: *Decoding Echo Chambers: LLM-Powered Simulations Revealing Polarization in Social Networks* (Coling 2025)  
> Chenxi Wang\*, Zongfang Liu\*, Dequan Yang, Xiuying Chen† — MBZUAI

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [File Structure](#file-structure)
3. [Setup](#setup)
4. [Running Simulations](#running-simulations)
5. [Output Format](#output-format)
6. [Metrics](#metrics)
7. [Key Concepts](#key-concepts)

---

## Project Overview

Three simulation modes, all using a **K-NN dynamic network** (each agent picks K most belief-similar neighbors per step):

| Mode | Script | Agent type | API needed |
|------|--------|-----------|------------|
| **Pure Numeric** | `numeric_dynamic/run_dynamic_numeric.py` | DynamicAgent (math rules) | No |
| **Pure LLM** | `llm_dynamic/run_dynamic_llm.py` | SocialAgent (GPT-4o-mini) | Yes |
| **Hybrid** | `hybrid_dynamic/run_dynamic_hybrid.py` | Mixed, ratio = alpha | Yes (if alpha < 1.0) |

**Hybrid alpha parameter:**
- `alpha=0.0` → 100% LLM agents
- `alpha=1.0` → 100% Numeric agents (DeGroot by default)
- `alpha=0.5` → 50% LLM + 50% Numeric

**Stage 1 results** (Scale-Free, K=5, 30 steps):

| alpha | LLM agents | Final PoA | Final Polarization | Final Modularity |
|-------|-----------|-----------|-------------------|-----------------|
| 0.0   | 50 | **1.277** (lowest) | 0.616 | 0.749 |
| 0.25  | 38 | 1.442 | 0.620 | 0.720 |
| 0.5   | 25 | 1.349 | 0.462 | 0.792 |
| 0.75  | 12 | 1.469 | 0.378 | 0.813 |
| 1.0   | 0  | 1.590 | 0.082 | 0.794 |

---

## File Structure

```
EchoChamberSim/
│
├── .env                              # Put OPENAI_API_KEY here (not committed)
├── .gitignore
│
│── Core (shared)
├── agent.py                          # SocialAgent: LLM agent with long/short-term memory
├── model.py                          # World: legacy LLM-only model (used by main.py)
├── network.py                        # Network generation (scale-free, small-world, random)
├── utils.py                          # OpenAI client, metrics, visualization helpers
├── prompt.py                         # GPT prompt templates (memory, opinion update)
├── visualization.py                  # Visualization utilities
├── main.py                           # Legacy entry point (static network, LLM only)
│
├── data/
│   ├── agents_backgrounds_num_agents_50_euthanasia_gpt-4o-mini.json   # Agent personas (name, age, traits)
│   ├── belief_keywords.json                                            # Keywords for belief scale {-2..+2}
│   ├── mitigation_perspectives.json                                    # Counter-narratives injected into agents
│   ├── numeric_sim_opnions_and_stubbornness_num_agents_50.json        # Initial beliefs & stubbornness ρ_i
│   ├── scale_free_network_num_agents_50_seed_50.json                  # Pre-generated static graph
│   ├── small_world_network_num_agents_50_seed_50.json
│   └── random_network_num_agents_50_seed_50.json
│
├── numeric_dynamic/                  # Pure numeric simulation (no API needed)
│   ├── run_dynamic_numeric.py        # Entry point
│   ├── dynamic_numeric_model.py      # DynamicWorld: K-NN rewiring + opinion update
│   ├── dynamic_numeric_agent.py      # DynamicAgent: DeGroot / FJ / BCM methods
│   ├── plot_dynamic_metrics.py       # Plot PoA, polarization, modularity over steps
│   └── generate_dynamic_gif.py       # Animate belief evolution on network
│
├── hybrid_dynamic/                   # Hybrid LLM + Numeric simulation
│   ├── run_dynamic_hybrid.py         # Entry point (--alpha controls LLM/Numeric ratio)
│   ├── dynamic_hybrid_model.py       # HybridDynamicWorld: mixes SocialAgent + DynamicAgent
│   └── plot_hybrid_metrics.py        # Compare metrics across alpha values
│
├── llm_dynamic/                      # Pure LLM simulation
│   ├── run_dynamic_llm.py            # Entry point
│   └── dynamic_llm_model.py          # DynamicLLMWorld: all SocialAgent with K-NN
│
├── experiments_dynamic/              # Output: numeric experiment results
│   └── {network_type}/euthanasia/
│       └── agents_50_{method}_K_{K}_sigma_{sigma}_seed_{seed}/
│           ├── model_overview.json           # Metrics per step (polarization, Q, PoA, ...)
│           ├── agents_data.json              # Per-agent belief trajectory
│           ├── edges_per_step.json           # K-NN graph edges at each step
│           └── agents_interaction_data.json
│
├── experiments_hybrid/               # Output: hybrid experiment results
│   └── {network_type}/euthanasia/
│       └── agents_50_K_{K}_alpha_{alpha}_sigma_{sigma}_seed_{seed}/
│           └── [same files as experiments_dynamic]
│
├── experiments_dynamic_llm/          # Output: pure LLM experiment results
│
├── experiments_gpt-4o-mini_formal/   # Output: legacy LLM experiments
├── experiments_gpt-4o-mini_mitigation_formal_v2/
│
├── Opinion Dynamic/                  # Charts and figures (auto-generated)
│   ├── DeGroot/                      # PNG charts for DeGroot method
│   ├── FJ/                           # PNG charts for Friedkin-Johnsen
│   ├── BCM/                          # PNG charts for Bounded Confidence Model
│   ├── Method Comparison/            # Cross-method comparison charts
│   ├── Hybrid/                       # Alpha-sweep comparison charts
│   ├── GIF/                          # Opinion dynamics GIFs per network × K
│   └── _legacy/                      # Old charts before reorganization
│
└── scripts/
    └── run_experiments.sh            # Batch shell script for all simulations
```

---

## Setup

### 1. Install dependencies

```bash
pip install mesa openai pydantic networkx matplotlib seaborn scipy python-louvain names-dataset tqdm
```

### 2. Configure API key

Edit `.env` in the project root:

```
OPENAI_API_KEY=sk-proj-your-key-here
```

The key is read automatically at startup. Only needed for LLM or Hybrid modes (alpha < 1.0). Pure numeric runs work without any key.

---

## Running Simulations

### Pure Numeric (no API key needed)

```bash
# Single run
python numeric_dynamic/run_dynamic_numeric.py \
  --network_type scale_free --K 5 --update_method degroot \
  --sigma 0.5 --num_agents 50 --step_count 30 --seed 50

# Sweep all methods (degroot, fj, bcm) × all networks × K=[3,5,10,20,49]
python numeric_dynamic/run_dynamic_numeric.py --run_all_methods --run_all_networks --run_all_K
```

Available `--update_method` options:
- `degroot` — z_i = mean(neighbors)
- `fj` — z_i = (1-ρ)·mean(neighbors) + ρ·s_i  (stubbornness pulls toward initial belief)
- `bcm` — z_i = mean(neighbors within ±ε); use `--epsilon` to set threshold (default: 0.3)

### Hybrid LLM + Numeric (API key required for alpha < 1.0)

```bash
# Single run (50% LLM, 50% Numeric)
python hybrid_dynamic/run_dynamic_hybrid.py \
  --network_type scale_free --K 5 --alpha 0.5 \
  --sigma 0.5 --num_agents 50 --step_count 30 --seed 50

# Sweep all alpha values
python hybrid_dynamic/run_dynamic_hybrid.py --run_all_alpha
```

### Pure LLM (API key required)

```bash
python llm_dynamic/run_dynamic_llm.py \
  --network_type scale_free --K 5 \
  --num_agents 50 --step_count 30 --seed 50
```

### Generate Charts

```bash
# Numeric method comparison charts
python numeric_dynamic/plot_dynamic_metrics.py

# Hybrid alpha comparison charts
python hybrid_dynamic/plot_hybrid_metrics.py
```

Charts are saved to `Opinion Dynamic/` (organized by method).

---

## Output Format

Each experiment saves to `experiments_{type}/{network}/euthanasia/{exp_name}/`:

### `model_overview.json` (one JSON object per line, one line per step)

```json
{"step": 1, "polarization": 0.462, "modularity": 0.792, "echo_chamber_effect": 0.831,
 "social_cost": 63.4, "poa": 1.349, "avg_belief": -0.12, "std_belief": 0.68, ...}
```

### `agents_data.json`

```json
{
  "0": {"initial_belief": -1, "beliefs": [-1, -0.8, -0.6, ...], "stubbornness": 0.3},
  "1": {"initial_belief": 1, "beliefs": [1, 0.9, ...], ...}
}
```

### `edges_per_step.json`

```json
{"step_0": [[0,3],[0,7],...], "step_1": [[0,2],...], ...}
```

---

## Metrics

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Polarization** Pz | (1/N) Σ(z_i − z̄)² | Variance of beliefs; higher = more divided |
| **Modularity** Q | Louvain algorithm | Community structure; higher = clearer echo chambers |
| **Echo Chamber Effect** | mean Gaussian similarity of K-NN neighbors | How similar each agent's neighbors are |
| **Social Cost** | Σ[ Σ_j w_ij(z_i−z_j)² + ρ·K·(z_i−s_i)² ] | Disagreement cost across whole network |
| **PoA** | Social Cost / Optimal Cost | 1.0 = socially optimal; higher = worse than Nash |

**Optimal Cost** is the Nash equilibrium on the *static* initial network — computed once at initialization and used as the PoA denominator throughout the simulation.

---

## Key Concepts

### K-NN Dynamic Network
Each step, every agent rewires its K outgoing edges to the K agents with most similar beliefs, weighted by a Gaussian similarity function `g(O_i, O_j) = exp(−(O_i−O_j)² / 2σ²)`. This creates opinion-homophily — the network becomes more polarized as beliefs cluster.

### Belief Scale
- **Numeric agents:** continuous `[-1.0, 1.0]`
- **LLM agents:** discrete `{-2, -1, 0, 1, 2}`, normalized to `[-1.0, 1.0]` for K-NN and metric computation
- **Stubbornness** ρ_i ∈ [0,1]: how strongly an agent sticks to its initial belief (used in FJ model)

### Agent Types
- **SocialAgent** (`agent.py`): Uses GPT-4o-mini. Maintains long-term and short-term memory. Each step makes 3 API calls: short memory update + long memory update + opinion update.
- **DynamicAgent** (`numeric_dynamic/dynamic_numeric_agent.py`): Pure math, no API calls. Supports DeGroot, FJ, and BCM update rules.

### Data Files
- Agent backgrounds and initial opinions for 50 agents on the "euthanasia" topic are pre-generated in `data/`. To run a different topic, regenerate these files.
- Network graphs (scale-free with Barabási–Albert, small-world with Watts–Strogatz, random Erdős–Rényi) are pre-generated with seed=50 for reproducibility.
