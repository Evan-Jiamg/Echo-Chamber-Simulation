"""
convergence.py — graph-structural convergence for the K-NN rewiring dynamics.

The object of convergence here is the graph G(t), not the opinion vector. The
system does not reach a fixed point: edges keep churning indefinitely. What does
settle is the *rate* of structural change, so convergence is defined as entry
into an attractor, detected on published graph-comparison measures.

Measures
--------
C_out(t), C_in(t)   Temporal correlation coefficient — the average fraction of a
                    node's neighbours retained between consecutive snapshots.
                    Tang, Scellato, Musolesi, Mascolo & Latora, "Small-world
                    behavior in time-varying graphs", Phys. Rev. E 81, 055101(R)
                    (2010); directed adaptation from Büttner, Salau & Krieter,
                    SpringerPlus 5:1198 (2016).

                        C_i^out(t) = |N_i^out(t) ∩ N_i^out(t+1)|
                                     / sqrt( d_i^out(t) · d_i^out(t+1) )
                        C^out(t)   = (1/A^out) Σ_i C_i^out(t)

                    with A^out the number of nodes of nonzero out-degree. With
                    K-NN rewiring every out-degree equals K, so C^out is exactly
                    the mean fraction of retained neighbours.

dS(t)               Normalised Hamming distance on edge sets:
                        dS(t) = |E(t) △ E(t-1)| / (|E(t)| + |E(t-1)|)

dL(t)               Spectral distance on the normalised Laplacian
                    L = I − D^{-1/2} A_u D^{-1/2} of the undirected projection:
                        dL(t) = || λ(t) − λ(t-1) ||_2 ,  λ sorted ascending

DeltaCon(t)         Principled graph similarity via Fast Belief Propagation.
                    Koutra, Vogelstein & Faloutsos, SDM 2013; extended in ACM
                    TKDD 10(3) (2016).
                        S = [ I + ε²D − εA ]^{-1}
                        d = sqrt( Σ_ij ( sqrt(s_ij^1) − sqrt(s_ij^2) )² )
                        sim = 1 / (1 + d)

Attractor
---------
fixed_point     E(t) = E(t-1) persistently
limit_cycle:p   E(t) = E(t-p) persistently, 2 ≤ p ≤ P_MAX
plateau         C_out has stopped changing but no exact period was found
none            neither, within T_max

Edge-set periodicity was validated against state-level recurrence
(min_p max_t ||z(t) − z(t-p)||_inf) on 180 Type-C runs: the two agree on 29-30
of every 30 runs, so the cheap edge-set test is used.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

# W and eps_C were calibrated on 90 Type-C runs (3 networks x 30 seeds) per
# topic at N=50, K=5, where W=10 / eps_C=0.01 fires for 90/90 runs with t_conv
# p95 of 28 (gun_control) and 33 (abortion).
#
# T_max, however, could not be taken from Type-C. Type-L positions come from
# scoring free text rather than a closed-form update, and the rewiring churn
# they produce is far higher: C_out plateaus near 0.98 for Type-C but only
# 0.83-0.87 at alpha=0, i.e. ~17% of each agent's neighbours turn over every
# step against ~2%. Convergence is correspondingly later — measured t_conv of
# 45 and 79 at alpha=0 versus 24-31 for Type-C — so the Type-C cap of 60 would
# have truncated pure-Type-L runs before they reached their attractor. T_max is
# set to 120 to clear the observed maximum with margin. It is only a safety cap:
# dynamic stopping ends each run at t_conv + post_window, so raising it costs
# nothing for runs that converge early.
DEFAULTS = dict(
    W=10,            # window length for the plateau test
    eps_C=0.01,      # relative change in mean C_out below which C is flat
    patience=5,      # consecutive steps the plateau test must hold
    post_window=10,  # steps to keep running after t_conv, to show the plateau
    T_max=120,       # hard cap; see note above
    P_MAX=20,        # largest period searched
)


def temporal_correlation(prev: dict[int, set], cur: dict[int, set]) -> float:
    """C^out(t): mean fraction of retained out-neighbours (Tang et al. 2010)."""
    vals = []
    for i, cur_i in cur.items():
        prev_i = prev.get(i, set())
        denom = np.sqrt(len(prev_i) * len(cur_i))
        if denom > 0:
            vals.append(len(prev_i & cur_i) / denom)
    return float(np.mean(vals)) if vals else 0.0


def hamming(prev_edges: frozenset, cur_edges: frozenset) -> float:
    """Normalised Hamming distance on edge sets."""
    tot = len(prev_edges) + len(cur_edges)
    return len(prev_edges ^ cur_edges) / tot if tot else 0.0


def laplacian_spectrum(A: np.ndarray) -> np.ndarray:
    """Sorted eigenvalues of the normalised Laplacian of the undirected projection."""
    Au = ((A + A.T) > 0).astype(float)
    d = Au.sum(1)
    inv = np.where(d > 0, 1.0 / np.sqrt(np.where(d > 0, d, 1.0)), 0.0)
    L = np.eye(len(Au)) - inv[:, None] * Au * inv[None, :]
    return np.sort(np.linalg.eigvalsh(L))


def _fabp(A: np.ndarray) -> np.ndarray:
    Au = ((A + A.T) > 0).astype(float)
    deg = Au.sum(1)
    eps = 1.0 / (1.0 + deg.max()) if deg.max() > 0 else 1.0
    return np.linalg.inv(np.eye(len(Au)) + eps * eps * np.diag(deg) - eps * Au)


def deltacon(A_prev: np.ndarray, A_cur: np.ndarray) -> float:
    """DeltaCon similarity in [0, 1] (Koutra et al., SDM 2013)."""
    s1, s2 = _fabp(A_prev), _fabp(A_cur)
    d = np.sqrt(np.sum((np.sqrt(np.abs(s1)) - np.sqrt(np.abs(s2))) ** 2))
    return float(1.0 / (1.0 + d))


class ConvergenceDetector:
    """Tracks graph-structural convergence and decides when to stop."""

    def __init__(self, **overrides):
        self.cfg = {**DEFAULTS, **overrides}
        self.C: list[float] = []            # C_out per step
        self._edge_hashes: list[int] = []
        self._edges: list[frozenset] = []
        self._plateau_streak = 0
        self.t_conv: int | None = None
        self.attractor: str | None = None
        self.period: int | None = None

    # ── per-step ────────────────────────────────────────────────────────────
    def observe(self, edges: frozenset, c_out: float | None) -> None:
        self._edges.append(edges)
        self._edge_hashes.append(hash(edges))
        if c_out is not None:
            self.C.append(c_out)
        if self.t_conv is None:
            self._detect()

    # ── detection ───────────────────────────────────────────────────────────
    def _detect(self) -> None:
        t = len(self._edges)

        # 1) exact recurrence: fixed point (p=1) or limit cycle (p>=2)
        p = self._period()
        if p is not None:
            self.t_conv = t
            self.period = p
            self.attractor = "fixed_point" if p == 1 else f"limit_cycle:{p}"
            return

        # 2) plateau in C_out
        W, eps, pat = self.cfg["W"], self.cfg["eps_C"], self.cfg["patience"]
        if len(self.C) >= 2 * W:
            a = float(np.mean(self.C[-W:]))
            b = float(np.mean(self.C[-2 * W:-W]))
            rel = abs(a - b) / b if b > 1e-12 else 0.0
            self._plateau_streak = self._plateau_streak + 1 if rel < eps else 0
            if self._plateau_streak >= pat:
                self.t_conv = t - pat + 1
                self.attractor = "plateau"

    def _period(self) -> int | None:
        """Smallest p with E(t) = E(t-p) held over the whole verification span."""
        h = self._edge_hashes
        n = len(h)
        for p in range(1, min(self.cfg["P_MAX"], n // 2) + 1):
            span = range(n - 2 * p, n - p)          # non-empty by construction
            if all(h[i] == h[i + p] for i in span) and \
               all(self._edges[i] == self._edges[i + p] for i in span):
                return p
        return None

    # ── stopping ────────────────────────────────────────────────────────────
    def should_stop(self, step: int) -> bool:
        if step >= self.cfg["T_max"]:
            return True
        return self.t_conv is not None and step >= self.t_conv + self.cfg["post_window"]

    def summary(self) -> dict:
        return {
            "t_conv": self.t_conv,
            "attractor": self.attractor or "none",
            "period": self.period,
            "steps_run": len(self._edges),
            "C_plateau": float(np.mean(self.C[-self.cfg["W"]:])) if self.C else None,
            "hit_T_max": self.t_conv is None,
        }


# ── Modularity null model ───────────────────────────────────────────────────
# Q has no absolute scale: random graphs carry nonzero modularity purely from
# fluctuations (Guimerà, Sales-Pardo & Amaral, Phys. Rev. E 70, 025101(R), 2004),
# and K-NN rewiring fixes every out-degree at K, a constraint that induces
# modularity on its own. Claims are therefore made on
#
#     Q_norm = Q_obs − mean_r Q_rand^(r)
#     z_Q    = ( Q_obs − mean_r Q_rand^(r) ) / sd_r Q_rand^(r)
#
# where G_rand preserves both the in- and out-degree sequence. Preserving both
# is what isolates structure beyond what the degree sequence already implies:
# out-degree is an exogenous constraint (=K), in-degree is the endogenous
# outcome (hub formation).

Q_NULL = dict(R=20, nswap_mult=10, max_tries_mult=1000)


def q_null(G, partition_fn, R=None, nswap_mult=None, max_tries_mult=None, seed=0):
    """Degree-preserving null distribution of modularity.

    Randomisation is performed on the *undirected projection*, because that is
    the graph modularity is computed on, using double-edge swaps that preserve
    its degree sequence.

    A directed three-cycle swap (`nx.directed_edge_swap`) was tried first and
    rejected: on this graph family it does not mix. Edge overlap with the
    original plateaus near 0.35 even at nswap = 50|E|, and the resulting Q_rand
    swings between 0.24 and 0.65 across consecutive steps whose degree sequences
    are near-identical — a null must not depend on its input that strongly.
    Undirected double-edge swaps mix to ~0.13 overlap and hold Q_rand to within
    0.01 across those same steps.

    Returns (mean, sd, n_failed). A randomisation that raises is counted in
    n_failed and dropped, never silently replaced by a configuration-model draw,
    which would not preserve the degree sequence exactly.
    """
    cfg = dict(Q_NULL)
    for k, v in (("R", R), ("nswap_mult", nswap_mult), ("max_tries_mult", max_tries_mult)):
        if v is not None:
            cfg[k] = v

    Gu = G.to_undirected() if G.is_directed() else G
    m = Gu.number_of_edges()
    if m == 0:
        return 0.0, 0.0, 0

    vals, failed = [], 0
    for r in range(cfg["R"]):
        H = Gu.copy()
        try:
            nx.double_edge_swap(H, nswap=cfg["nswap_mult"] * m,
                                max_tries=cfg["max_tries_mult"] * m, seed=seed + r)
        except Exception:
            failed += 1
            continue
        vals.append(partition_fn(H))

    if not vals:
        return float("nan"), float("nan"), failed
    return float(np.mean(vals)), float(np.std(vals)), failed
