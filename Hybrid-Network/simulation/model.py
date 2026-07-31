import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

import mesa
import json
import csv
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
import networkx as nx
import community as community_louvain

from numeric_agent import DynamicAgent   # numeric agent (from dynamic_network/numeric/)
from scorer import build_scorer
import convergence as conv
from agent import SocialAgent            # LLM agent (from dynamic_network/)
from utils import update_day             # GPT opinion update (from dynamic_network/)


def _louvain_q(G):
    """Modularity of the undirected projection — used for both G and its nulls."""
    Gu = G.to_undirected()
    if Gu.number_of_edges() == 0:
        return 0.0
    return community_louvain.modularity(community_louvain.best_partition(Gu), Gu)


def _ari(a, b):
    """Adjusted Rand index (Hubert & Arabie 1985)."""
    a, b = np.asarray(a), np.asarray(b)
    n = len(a)
    ua, ub = np.unique(a), np.unique(b)
    C = np.zeros((len(ua), len(ub)))
    for i, x in enumerate(ua):
        for j, y in enumerate(ub):
            C[i, j] = np.sum((a == x) & (b == y))
    c2 = lambda m: m * (m - 1) / 2
    sij, si, sj, tot = c2(C).sum(), c2(C.sum(1)).sum(), c2(C.sum(0)).sum(), c2(n)
    exp, mx = si * sj / tot, 0.5 * (si + sj)
    return 0.0 if mx == exp else float((sij - exp) / (mx - exp))


def _load_network(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    G = nx.Graph()
    G.add_nodes_from(data["nodes"])
    G.add_edges_from(data["edges"])
    return G


class HybridDynamicWorld(mesa.Model):
    """
    Hybrid Dynamic Network:
      alpha=0 -> pure LLM  (all SocialAgent, GPT-driven)
      alpha=1 -> pure Numeric (all DynamicAgent, DeGroot)
      0 < alpha < 1 -> mixed

    Alpha > 0 requires LLM API key in utils.py and agent backgrounds file.
    Alpha = 1 runs without any API requirement.

    Belief scales:
      Numeric  : float [-1, 1]  (canonical as-is)
      LLM      : int {-2,-1,0,1,2} / 2 = canonical [-1,-0.5,0,0.5,1]
    All metrics use canonical beliefs in [-1, 1].
    """

    def __init__(self, network_type, K, alpha=1.0, num_agents=50, seed=50,
                 topic="euthanasia", gpt_model="mistral-nemo-instruct-2407",
                 belief_keywords_file=None, exp_dir=None,
                 leaders=None, temp=0.0, with_long_memory=True,
                 backgrounds_label="gpt-4o-mini", field_order="cot_first",
                 scorer_kind="self_report", scorer_kwargs=None, conv_cfg=None,
                 qnull_cfg=None, max_workers=50):
        super().__init__()

        if leaders is None:
            leaders = [10, 30]
        if belief_keywords_file is None:
            belief_keywords_file = os.path.join(
                os.path.dirname(__file__), '..', 'data', 'lexicons',
                'belief_keywords.json')
        if exp_dir is None:
            exp_dir = os.path.join(
                os.path.dirname(__file__), '..', 'experiments')

        self.network_type = network_type
        self.K = K
        self.alpha = alpha
        self.num_agents = num_agents
        self.leaders = leaders
        self.gpt_model = gpt_model
        self.temp = temp
        self.with_long_memory = with_long_memory
        self.backgrounds_label = backgrounds_label
        self.field_order = field_order
        self.scorer = build_scorer(scorer_kind, **(scorer_kwargs or {}))
        self.parse_failures = 0
        self.step_parse_fail = 0
        self.llm_calls = 0
        self.current_step = 0
        self._tb_rng = random.Random(seed)   # K-NN tie-break only
        self.qnull_cfg = qnull_cfg
        self.max_workers = int(max_workers)
        self._qnull_seed = seed * 100003
        self.detector = conv.ConvergenceDetector(**(conv_cfg or {}))
        self._prev_nbr = None
        self._prev_edges = None
        self._prev_A = None
        self._prev_spec = None

        # Path mirrors the experiment matrix: every dimension that can vary
        # while the others are held fixed gets its own level, so two runs
        # differing in any one of them can never share a directory. N and K
        # are fixed for the campaign and live in the run manifest instead.
        self.run_dir = os.path.join(
            exp_dir, topic, network_type,
            f"alpha_{alpha:.3f}", f"seed_{seed:02d}")
        os.makedirs(self.run_dir, exist_ok=True)

        # Load beliefs and stubbornness — prefer topic-specific file if it exists
        _topic_data = os.path.join(
            os.path.dirname(__file__), '..', 'data', 'agents',
            f'numeric_sim_opnions_and_stubbornness_num_agents_{num_agents}_{topic}.json'
        )
        _default_data = os.path.join(
            os.path.dirname(__file__), '..', 'data', 'agents',
            f'numeric_sim_opnions_and_stubbornness_num_agents_{num_agents}.json'
        )
        data_file = _topic_data if os.path.exists(_topic_data) else _default_data
        print(f"[Init] Loading beliefs from: {os.path.basename(data_file)}")
        with open(data_file, 'r') as f:
            raw = json.load(f)
        self._init_beliefs_numeric = {int(k): float(v) for k, v in raw['opinions'].items()}
        self.stubbornness = {int(k): float(v) for k, v in raw['stubbornness'].items()}

        # Determine agent type assignment
        n_llm = round((1.0 - alpha) * num_agents)
        n_numeric = num_agents - n_llm

        if n_llm == 0:
            self.llm_agent_ids = set()
            self.numeric_agent_ids = set(range(num_agents))
        elif n_numeric == 0:
            self.llm_agent_ids = set(range(num_agents))
            self.numeric_agent_ids = set()
        else:
            # Leaders first in LLM pool, then fill randomly
            self.llm_agent_ids = set(leaders[:min(len(leaders), n_llm)])
            remaining = [i for i in range(num_agents) if i not in self.llm_agent_ids]
            random.shuffle(remaining)
            extra = n_llm - len(self.llm_agent_ids)
            if extra > 0:
                self.llm_agent_ids.update(remaining[:extra])
            self.numeric_agent_ids = set(range(num_agents)) - self.llm_agent_ids

        # Load initial static network
        network_file = os.path.join(
            os.path.dirname(__file__), '..', 'data', 'networks',
            f'{network_type}_network_num_agents_{num_agents}_seed_{seed}.json'
        )
        self.G_static = _load_network(network_file)

        # Load LLM backgrounds only if needed
        self.backgrounds = {}
        self.topic_key = topic
        self.topic_str = ""
        self.belief_keywords = {}
        if self.llm_agent_ids:
            backgrounds_file = os.path.join(
                os.path.dirname(__file__), '..', 'data', 'agents',
                f'agents_backgrounds_num_agents_{num_agents}_{topic}_{backgrounds_label}.json'
            )
            if not os.path.exists(backgrounds_file):
                raise FileNotFoundError(
                    f"Backgrounds not found: {backgrounds_file}\n"
                    "Run main.py once to generate backgrounds, or use alpha=1.0 (pure numeric)."
                )
            with open(backgrounds_file, 'r') as f:
                self.backgrounds = json.load(f)["backgrounds"]
            opinions_file = os.path.join(os.path.dirname(__file__), '..', 'data',
                                         'lexicons', 'opinions.json')
            with open(opinions_file, 'r') as f:
                self.topic_str = json.load(f)[topic]
            with open(belief_keywords_file, 'r') as f:
                self.belief_keywords = json.load(f)

        # Create agents
        self.agents_list = []
        for i in tqdm(range(num_agents), desc=f"Creating agents (LLM={n_llm} Num={n_numeric})"):
            agent = self._create_llm_agent(i) if i in self.llm_agent_ids \
                else self._create_numeric_agent(i)
            self.agents_list.append(agent)

        # Initial neighbors from static network
        self.neighbors = {
            i: list(self.G_static.neighbors(i))
            for i in range(num_agents)
        }

        # DiGraph: updated by K-NN each step
        self.G = nx.DiGraph()
        self.G.add_nodes_from(range(num_agents))
        for i, nbrs in self.neighbors.items():
            for j in nbrs:
                self.G.add_edge(i, j)

        self.optimal_cost = self._compute_optimal_cost()
        self.edges_log = []

        # Log agent type assignment for reference
        assignment = {
            "llm_agents": sorted(self.llm_agent_ids),
            "numeric_agents": sorted(self.numeric_agent_ids),
            "alpha": alpha,
            "n_llm": n_llm,
            "n_numeric": n_numeric,
        }
        with open(os.path.join(self.run_dir, "agent_assignment.json"), 'w') as f:
            json.dump(assignment, f, indent=4)

    # ── Agent factories ────────────────────────────────────────────────────
    def _create_numeric_agent(self, i):
        return DynamicAgent(
            unique_id=i,
            model=self,
            initial_belief=self._init_beliefs_numeric[i],
            stubbornness=self.stubbornness[i]
        )

    def _create_llm_agent(self, i):
        # The persona moves into the user message and leadership becomes a flag,
        # so every agent's system message is byte-identical and vLLM can reuse
        # its KV cache instead of re-prefilling the shared instructions once per
        # agent per step.
        bg = self.backgrounds[str(i)]
        return SocialAgent(
            model=self,
            unique_id=i,
            name=bg["name"],
            gender=bg["gender"],
            age=bg["age"],
            traits=bg["traits"],
            qualification=bg["education level"],
            initial_belief=self._init_beliefs_numeric[i],
            topic=self.topic_str,
            gpt_model=self.gpt_model,
            temp=self.temp,
            initial_opinion=bg.get("initial_opinion"),
            initial_reasoning=bg.get("initial_reasoning"),
            is_leader=(i in self.leaders),
            field_order=self.field_order,
        )

    # ── Canonical belief: all agents unified to [-1, 1] ───────────────────
    def _canonical(self, i):
        a = self.agents_list[i]
        return float(a.belief)

    def _initial_canonical(self, i):
        return self._init_beliefs_numeric[i]

    # ── Optimal cost (Central Planner minimum, on current K-NN DiGraph) ──────
    def _compute_optimal_cost(self):
        # Social cost: C(z) = Σ_{i→j}(z_i-z_j)² + Σ_i ρ_i·K·(z_i-s_i)²
        #                    = z^T L_sym z + (z-s)^T W (z-s)
        # where L_sym = D_out + D_in − A − A^T  (equals Σ_{i→j}(z_i-z_j)² in matrix form)
        # ∂C/∂z = 0  →  (L_sym + W)·z_opt = W·s
        nodes = list(range(self.num_agents))
        A = nx.to_numpy_array(self.G, nodelist=nodes)
        D_out = np.diag(A.sum(axis=1))
        D_in  = np.diag(A.sum(axis=0))
        L_sym = D_out + D_in - A - A.T
        rho = np.array([self.stubbornness[i] for i in nodes])
        s   = np.array([self._initial_canonical(i) for i in nodes])
        W   = self.K * np.diag(rho)
        M   = L_sym + W
        if np.linalg.matrix_rank(M) < len(nodes):
            M += 1e-8 * np.eye(len(nodes))
        z_opt = np.linalg.solve(M, W @ s)
        dz = z_opt[:, None] - z_opt[None, :]
        cost = float(np.sum(A * dz ** 2))
        cost += float(np.dot(rho * self.K, (z_opt - s) ** 2))
        return cost

    # ── K-NN rewiring ──────────────────────────────────────────────────────
    def _update_knn(self):
        # Ties at the K-th cutoff are broken at random, with a dedicated seeded
        # RNG. Python's sort is stable, so ranking a fixed 0..N-1 candidate list
        # would hand every tie to the lowest agent index — under a coarse
        # opinion scale nearly every selection is a tie, so hub identity would
        # be an artefact of numbering rather than of opinion.
        canonical = [self._canonical(i) for i in range(self.num_agents)]
        intrinsic = [self._initial_canonical(i) for i in range(self.num_agents)]
        order = list(range(self.num_agents))
        for i in range(self.num_agents):
            self._tb_rng.shuffle(order)
            dists = sorted(
                [(j, abs(intrinsic[i] - canonical[j])) for j in order if j != i],
                key=lambda x: x[1]
            )
            self.neighbors[i] = [j for j, _ in dists[:self.K]]
        self.G.remove_edges_from(list(self.G.edges()))
        for i, nbrs in self.neighbors.items():
            for j in nbrs:
                self.G.add_edge(i, j)

    # ── Main step ──────────────────────────────────────────────────────────
    def step(self):
        self.step_parse_fail = 0
        canonical_snapshot = [self._canonical(i) for i in range(self.num_agents)]

        # LLM agents: memory interaction + GPT belief update
        if self.llm_agent_ids:
            for i in self.llm_agent_ids:
                self.agents_list[i].agent_interaction = [
                    self.agents_list[j] for j in self.neighbors[i]
                ]
            shuffled_llm = [self.agents_list[i] for i in self.llm_agent_ids]
            random.shuffle(shuffled_llm)
            # Concurrency cap. vLLM batches across all in-flight requests, so
            # what matters is keeping roughly max_num_seqs of them outstanding.
            # Configurable because the best split between processes and threads
            # is measured (T-0), not assumed.
            n_workers = min(len(shuffled_llm), self.max_workers)

            # Phase 1: interact() — all agents read from previous-step opinions
            # Safe to parallelize: reads opinions[-1] (pre-update), writes only to self
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = [pool.submit(agent.step) for agent in shuffled_llm]
                wait(futures, return_when=ALL_COMPLETED)
                for f in futures:
                    f.result()

            # Phase 2: one LLM call per agent — new opinion, reasoning, memory.
            # Safe to parallelize: reads/writes only own state; K-NN runs after this barrier
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = [pool.submit(update_day, agent) for agent in shuffled_llm]
                wait(futures, return_when=ALL_COMPLETED)
                for f in futures:
                    f.result()

            # Phase 3: batched continuous stance scoring. One forward pass for
            # the whole step, on the same scale as the Reddit corpus.
            texts = [a.opinions[-1] for a in shuffled_llm]
            for agent, stance in zip(shuffled_llm, self.scorer.score(texts)):
                agent.set_belief(stance)
                self.llm_calls += 1
                if agent.parse_failed:
                    self.parse_failures += 1
                    self.step_parse_fail += 1

            # A whole step failing means the LLM endpoint is broken — engine
            # dead, port taken by a stale server, model not loaded — not that
            # every agent independently chose to hold its opinion. Fail loudly
            # rather than let a multi-day sweep fill with frozen runs.
            if self.step_parse_fail == len(shuffled_llm):
                self._dead_steps = getattr(self, "_dead_steps", 0) + 1
                if self._dead_steps >= 2:
                    raise RuntimeError(
                        f"LLM endpoint appears dead: {self._dead_steps} consecutive "
                        f"steps with {len(shuffled_llm)}/{len(shuffled_llm)} parse "
                        f"failures. Check the vLLM server before resuming.")
            else:
                self._dead_steps = 0

        # Numeric agents: synchronous FJ on canonical snapshot
        for i in self.numeric_agent_ids:
            nbr_beliefs = [canonical_snapshot[j] for j in self.neighbors[i]]
            self.agents_list[i].step_fj(nbr_beliefs)
            self.agents_list[i].record()

        # K-NN rewiring based on updated beliefs
        self._update_knn()
        self.optimal_cost = self._compute_optimal_cost()
        self._observe_structure()
        self.edges_log.append([list(e) for e in self.G.edges()])
        self.current_step += 1
        self.save_model_data()
        self.save_agents_data()


    # ── Graph-structural convergence ────────────────────────────────────────
    def _observe_structure(self):
        """Compute the graph-comparison measures for this step and feed the
        convergence detector. All measures are defined in core/convergence.py."""
        cur_nbr = {i: set(v) for i, v in self.neighbors.items()}
        cur_edges = frozenset((i, j) for i, ns in cur_nbr.items() for j in ns)
        A = nx.to_numpy_array(self.G, nodelist=list(range(self.num_agents)))
        spec = conv.laplacian_spectrum(A)

        if self._prev_nbr is None:
            self.step_metrics = dict(dS=None, C_out=None, dL=None, deltacon=None)
            c_out = None
        else:
            c_out = conv.temporal_correlation(self._prev_nbr, cur_nbr)
            self.step_metrics = dict(
                dS=conv.hamming(self._prev_edges, cur_edges),
                C_out=c_out,
                dL=float(np.linalg.norm(spec - self._prev_spec)),
                deltacon=conv.deltacon(self._prev_A, A),
            )

        self.detector.observe(cur_edges, c_out)
        self._prev_nbr, self._prev_edges = cur_nbr, cur_edges
        self._prev_A, self._prev_spec = A, spec

    def _structure_stats(self):
        """Community and hub descriptors plus the modularity null model."""
        G_u = self.G.to_undirected()
        if G_u.number_of_edges() == 0:
            return dict(n_comm=0, max_comm_share=0.0, ari_camp=0.0, in_gini=0.0,
                        Q_rand_mean=float("nan"), Q_rand_sd=float("nan"),
                        Q_norm=float("nan"), z_Q=float("nan"), swap_fail=0)
        part = community_louvain.best_partition(G_u)
        lab = np.array([part[i] for i in range(self.num_agents)])
        camp = np.array([1 if self._initial_canonical(i) > 0 else 0
                         for i in range(self.num_agents)])
        sizes = np.bincount(lab)
        ind = np.array([self.G.in_degree(i) for i in range(self.num_agents)], float)
        srt = np.sort(ind); n = len(srt)
        gini = (float((2 * np.arange(1, n + 1) - n - 1).dot(srt) / (n * srt.sum()))
                if srt.sum() > 0 else 0.0)

        q_obs = community_louvain.modularity(part, G_u)
        q_mean, q_sd, n_fail = conv.q_null(
            self.G, _louvain_q, seed=self._qnull_seed, **(self.qnull_cfg or {}))
        self._qnull_seed += 1000
        z = (q_obs - q_mean) / q_sd if q_sd and q_sd > 0 else float("nan")

        return dict(n_comm=len(sizes),
                    max_comm_share=float(sizes.max() / self.num_agents),
                    ari_camp=_ari(camp, lab),
                    in_gini=gini,
                    Q_rand_mean=q_mean, Q_rand_sd=q_sd,
                    Q_norm=q_obs - q_mean, z_Q=z, swap_fail=n_fail)

    # ── Metrics ───────────────────────────────────────────────────────────
    def compute_total_social_cost(self):
        total = 0.0
        for i in range(self.num_agents):
            zi = self._canonical(i)
            si = self._initial_canonical(i)
            nbr = [self._canonical(j) for j in self.neighbors[i]]
            total += sum((zi - b) ** 2 for b in nbr)
            total += self.stubbornness[i] * self.K * (zi - si) ** 2
        return float(total)

    def compute_polarization(self):
        z = np.array([self._canonical(i) for i in range(self.num_agents)])
        return float(np.mean((z - z.mean()) ** 2))

    def compute_modularity(self):
        G_u = self.G.to_undirected()
        if G_u.number_of_edges() == 0:
            return 0.0
        partition = community_louvain.best_partition(G_u)
        return float(community_louvain.modularity(partition, G_u))

    _CSV_FIELDS = ["step", "polarization", "modularity", "poa",
                   "dS", "C_out", "dL", "deltacon",
                   "n_comm", "max_comm_share", "ari_camp", "in_gini",
                   "parse_fail",
                   "Q_rand_mean", "Q_rand_sd", "Q_norm", "z_Q", "swap_fail"]

    def save_model_data(self):
        cost = self.compute_total_social_cost()
        poa = cost / self.optimal_cost if self.optimal_cost > 0 else 1.0
        struct = self._structure_stats()
        data = {
            "step": self.current_step,
            "polarization": self.compute_polarization(),
            "modularity": self.compute_modularity(),
            "poa": poa,
            "parse_fail": self.step_parse_fail,
        }
        data.update(getattr(self, "step_metrics", {}) or
                    dict(dS=None, C_out=None, dL=None, deltacon=None))
        data.update(struct)
        with open(os.path.join(self.run_dir, "model_overview.json"), 'a') as f:
            json.dump(data, f)
            f.write('\n')
        csv_path = os.path.join(self.run_dir, "metrics.csv")
        write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self._CSV_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(data)

    def save_agents_data(self):
        data = {}
        for a in self.agents_list:
            entry = {"beliefs": a.beliefs, "opinions": a.opinions}
            if a.unique_id in self.llm_agent_ids:
                entry["reasonings"] = a.reasonings
                entry["short_memory"] = a.short_memory_full
                entry["long_memory"] = a.long_memory_full
            data[str(a.unique_id)] = entry
        with open(os.path.join(self.run_dir, "agents_data.json"), 'w') as f:
            json.dump(data, f, indent=4)

    def save_interaction_data(self):
        data = {
            str(a.unique_id): {"beliefs": a.beliefs, "opinions": a.opinions}
            for a in self.agents_list
        }
        with open(os.path.join(self.run_dir, "agents_interaction_data.json"), 'w') as f:
            json.dump(data, f, indent=4)

    def save_edges_data(self):
        with open(os.path.join(self.run_dir, "edges_per_step.json"), 'w') as f:
            json.dump(self.edges_log, f)

    def plot_run_metrics(self):
        csv_path = os.path.join(self.run_dir, "metrics.csv")
        if not os.path.exists(csv_path):
            return
        rows = []
        with open(csv_path, newline='') as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return

        steps = [int(r["step"]) for r in rows]
        metrics = [
            ("polarization", "Polarization $P_z$",  "$P_z$"),
            ("modularity",   "Modularity $Q$",        "$Q$"),
            ("poa",          "Price of Anarchy",      "PoA"),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        title = (f"Hybrid — {self.network_type}, K={self.K}, α={self.alpha}, "
                 f"agents={self.num_agents}")
        fig.suptitle(title, fontsize=12)
        for ax, (key, label, ylabel) in zip(axes, metrics):
            values = [float(r[key]) for r in rows]
            ax.plot(steps, values, linewidth=2, color="tab:blue")
            ax.set_title(label, fontsize=10)
            ax.set_xlabel("Step")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            if key == "poa":
                ax.axhline(1.0, color="black", linestyle=":", linewidth=1, alpha=0.6)
        plt.tight_layout()
        # Beside the run's own metrics.csv. These used to land in
        # analysis/figures/, which now holds the paper figures, so a
        # sweep would bury them under one chart per (network, alpha).
        charts_dir = self.run_dir
        os.makedirs(charts_dir, exist_ok=True)
        fname = (f"hybrid_{self.network_type}_K{self.K}_alpha{self.alpha}"
                 f"_agents{self.num_agents}.png")
        out = os.path.join(charts_dir, fname)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Chart saved : {out}")

    # ── Resume support ──────────────────────────────────────────────────────
    def _resume_or_reset(self):
        """Prepare output files for a fresh run or continue an interrupted one.

        A sweep of this size cannot afford to lose completed work to an
        interruption, so a run whose `convergence.json` already exists is
        treated as finished and skipped. A run with a partial `metrics.csv` but
        no `convergence.json` is restarted from scratch: the simulation state
        (opinions, graph, memories) is not checkpointed, so continuing from a
        partial trajectory would silently splice two different runs together.

        Returns True if the caller should run the simulation, False if this run
        is already complete.
        """
        conv_path = os.path.join(self.run_dir, "convergence.json")
        if os.path.exists(conv_path):
            print(f"[Resume] already complete, skipping: {self.run_dir}")
            return False

        csv_path = os.path.join(self.run_dir, "metrics.csv")
        if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
            n = sum(1 for _ in open(csv_path)) - 1
            print(f"[Resume] partial run ({n} steps, no convergence.json) — restarting")

        for fname in ("model_overview.json", "metrics.csv"):
            open(os.path.join(self.run_dir, fname), 'w').close()
        return True

    def _finish(self, cap, resumed_complete=False):
        if resumed_complete:
            return
        n_llm, n_num = len(self.llm_agent_ids), len(self.numeric_agent_ids)

        self.save_agents_data()
        self.save_interaction_data()
        self.save_edges_data()
        self.plot_run_metrics()

        summary = self.detector.summary()
        summary.update(alpha=self.alpha, network=self.network_type, topic=self.topic_key, topic_desc=getattr(self, "topic_str", None),
                       n_llm=n_llm, n_numeric=n_num,
                       parse_failures=self.parse_failures, llm_calls=self.llm_calls)
        with open(os.path.join(self.run_dir, "convergence.json"), "w") as f:
            json.dump(summary, f, indent=2)

        final_cost = self.compute_total_social_cost()
        print(f"\nalpha={self.alpha}  LLM={n_llm}  Numeric={n_num}")
        print(f"Stopped at   : step {self.current_step} / cap {cap}")
        print(f"Convergence  : t_conv={summary['t_conv']}  attractor={summary['attractor']}")
        print(f"C_out plateau: {summary['C_plateau']}")
        print(f"Final PoA    : {final_cost / self.optimal_cost:.4f}")
        print(f"Saved to     : {self.run_dir}")

    def run_model(self, step_count=None):
        """Run until the graph structure has converged, then a short post-window.

        `step_count` is only an upper bound; the effective stopping point is
        decided by core/convergence.py from the graph-comparison measures. A run
        that never converges stops at T_max and is flagged.
        """
        cap = step_count or self.detector.cfg["T_max"]
        if not self._resume_or_reset():
            return self._finish(cap, resumed_complete=True)
        bar = tqdm(total=cap, desc=f"Hybrid {self.network_type} K={self.K} alpha={self.alpha}")
        while self.current_step < cap:
            self.step()
            bar.update(1)
            if self.detector.should_stop(self.current_step):
                break
        bar.close()

        return self._finish(cap)
