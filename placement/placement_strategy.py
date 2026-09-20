"""
Meter-placement strategies for RES.md D10 Phase 4 sub-study 4A (meter-
density sweep x placement strategy). All four strategies share one
FeederContext (feeder_context.py) computed once from (network, loads).

HybridPlacement is the original HybridRepresentativePlacement, moved
here and defect-fixed (see class docstring). RandomPlacement,
DegreePlacement and CommunityDetectionPlacement (stub) are new.
"""

from abc import ABC, abstractmethod
from collections import defaultdict, deque
import json
import math
import random
from pathlib import Path

# RES.md D10 Phase 4 sub-study 4A: "Reference configuration: introduce
# it explicitly in config, defaulting to 49 meters (57.6%)... Nothing
# else may hardcode 49." This is the ONE place that reads the 49; every
# strategy constructor below defaults to it rather than hardcoding its
# own copy.
_REFERENCE_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "config" / "reference_configuration.json"
)
with open(_REFERENCE_CONFIG_PATH) as _f:
    REFERENCE_TARGET_METERS = json.load(_f)["target_meters"]


class PlacementStrategy(ABC):

    @abstractmethod
    def select(self, loads, context):
        """loads must already be annotated via context.annotate_loads()."""


class HybridPlacement(PlacementStrategy):
    """Moved from parser/placement_strategy.py's HybridRepresentativePlacement.
    Graph build, topology collapse, rooted tree, bus loads, downstream
    load, main feeder extraction, depths and branch IDs now live in
    FeederContext (computed once, shared across strategies) -- this
    class keeps only what is actually hybrid-specific: size-class
    quotas, scoring, round-robin branch selection, count top-up, and
    phase balancing -- the latter now INCREMENTAL (applied at every
    pick against running per-phase kW, not a post-hoc swap pass over
    the completed set -- see _pick_phase_aware). The swap-pass version
    did not survive nested construction (sweep_4a_tier1_report.txt
    FINDINGS #4); this fixes that. All candidate loads are also
    collapsed one-per-bus (_one_load_per_bus) before anything else runs,
    matching every other strategy's tie-break rule on the 3 multi-load
    buses (sweep_4a_tier1_report.txt FINDINGS #2).
    """

    # Score weights (Step-3 defect #7): kw / depth / downstream / backbone.
    # backbone is a genuine 0/1 indicator (on_backbone), not a 1.5x bonus --
    # the original weights (0.30/0.20/0.40/0.20) summed to 1.10 once the
    # 1.5x backbone bonus made its effective contribution 0.30, not 0.20,
    # pushing the achievable total to 1.20. Renormalised here by dividing
    # the original nominal weights by their sum (1.10), preserving their
    # original relative emphasis exactly, so every term is bounded to
    # [0,1] and the four weights sum to 1.00.
    _RAW_WEIGHTS = {"kw": 0.30, "depth": 0.20, "downstream": 0.40, "backbone": 0.20}
    _WEIGHT_SUM = sum(_RAW_WEIGHTS.values())
    WEIGHT_KW = _RAW_WEIGHTS["kw"] / _WEIGHT_SUM
    WEIGHT_DEPTH = _RAW_WEIGHTS["depth"] / _WEIGHT_SUM
    WEIGHT_DOWNSTREAM = _RAW_WEIGHTS["downstream"] / _WEIGHT_SUM
    WEIGHT_BACKBONE = _RAW_WEIGHTS["backbone"] / _WEIGHT_SUM

    # Phase-coverage baseline this placement balances against -- connected
    # kW per phase, not node count (RES.md D10 Phase 4 sub-study 4A: "Balancing
    # on kW is the better argument for detection (observe where the load is)").
    PHASE_KW_BASELINE = {"A": 1400.0, "B": 952.5, "C": 1137.5}

    METER_CEILING = 85  # distinct load buses (model verification: 85 of 91)
#if target_meters exceeds 85, raises a ValueError explaining why the ceiling exists (three buses share multiple loads).
    def __init__(self, target_meters=REFERENCE_TARGET_METERS, verbose=False):
        if target_meters > self.METER_CEILING:
            raise ValueError(
                f"target_meters={target_meters} exceeds the meter ceiling "
                f"({self.METER_CEILING} distinct load buses -- model "
                f"verification found 6 of 91 loads share buses: "
                f"S49a/b/c, S65a/b/c, S76a/b/c)."
            )

        self.target_meters = target_meters
        self.verbose = verbose

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def select(self, loads, context):
        # Sweep-defect #2 fix: the SAME tie-break rule every other
        # strategy already uses for multi-load buses (S49a/b/c,
        # S65a/b/c, S76a/b/c -- highest kW wins, via _one_load_per_bus)
        # now applies here too, BEFORE any Hybrid-specific logic sees
        # the candidate pool. Previously Hybrid could pick a different
        # sibling than Random/Degree/Community at the same bus (its
        # round-robin/score choice vs. their highest-kW rule), which
        # showed up as a spurious phase-composition difference even at
        # the 85-meter ceiling, where every strategy is forced onto the
        # same 85 buses (sweep_4a_tier1_report.txt FINDINGS #2).
        loads = _one_load_per_bus(loads)

        loads = self._assign_size_classes(loads)

        if self.verbose:
            print("\n===== SOURCE =====")
            print(context.network["source"])
            print("\n===== REGULATORS =====")
            for reg in context.network["assets"].get("regulators", []):
                print(reg)

        self._assign_scores(loads)

        selected = self._representative_selection(loads)

        selected = self._top_up_shortfall(selected, loads)

        assert len(selected) == self.target_meters, (
            f"selected {len(selected)} meters, expected {self.target_meters}"
        )
        assert len({l["bus"] for l in selected}) == len(selected), (
            "duplicate bus in selection"
        )

        if self.verbose:
            for load in selected:
                print(load["name"], load["bus"], load["branch"])

        return selected

    # --------------------------------------------------
    # Stage 1 -- size classes
    # --------------------------------------------------

    def _assign_size_classes(self, loads):
        kw = sorted(load["base_kw"] for load in loads)
        q1, q2, q3 = self._compute_quartiles(kw)

        for load in loads:
            value = load["base_kw"]

            if value <= q1:
                load["size_class"] = "Small"
            elif value <= q2:
                load["size_class"] = "Medium"
            elif value <= q3:
                load["size_class"] = "Large"
            else:
                load["size_class"] = "Very Large"

        return loads

    def _compute_quartiles(self, values):
        n = len(values)

        def percentile(p):
            index = (n - 1) * p
            lower = math.floor(index)
            upper = math.ceil(index)

            if lower == upper:
                return values[lower]

            fraction = index - lower
            return values[lower] + (values[upper] - values[lower]) * fraction

        return percentile(0.25), percentile(0.50), percentile(0.75)

    # --------------------------------------------------
    # Stage 2 -- scoring
    # --------------------------------------------------

    def _assign_scores(self, loads):
        max_depth = max(load["depth"] for load in loads)

        for load in loads:
            kw_score = min(load["base_kw"] / 100.0, 1.0)
            depth_score = load["depth"] / max_depth
            downstream_score = load["downstream_score"]
            backbone_indicator = 1.0 if load["on_backbone"] else 0.0

            load["score"] = (
                self.WEIGHT_KW * kw_score +
                self.WEIGHT_DEPTH * depth_score +
                self.WEIGHT_DOWNSTREAM * downstream_score +
                self.WEIGHT_BACKBONE * backbone_indicator
            )

    # --------------------------------------------------
    # Stage 3 -- quota-based round-robin selection, phase-aware
    # (sweep-defect #1 fix: phase balancing is now INCREMENTAL --
    # applied at every pick, using kW selected so far, rather than as
    # a post-hoc swap pass over the completed set. This makes the
    # selection order itself approximately phase-balanced at every
    # PREFIX length, not just at the final target_meters -- which is
    # what nested construction needs. See _pick_phase_aware below and
    # reports/sweep_4a_tier1_report.txt FINDINGS #4 for why the old
    # post-hoc pass didn't survive nesting.)
    # --------------------------------------------------

    def _representative_selection(self, loads):
        """Quotas are still computed per size class (coverage across
        size classes is preserved), but classes are now processed in
        GLOBAL round-robin turns -- one pick per class per round -- not
        one class exhausted fully before the next starts. Earlier
        versions processed class 1's entire quota, then class 2's, etc.,
        which meant a prefix shorter than class 1's quota reflected only
        class 1's phase mix, not a global balance -- this is what kept
        nested-vs-independent divergence large even after making the
        within-class pick phase-aware. Branch diversity within a class
        is preserved via a rotating per-class branch queue (see
        _pick_one_for_class), so successive turns for the same class
        still cycle through all of that class's branches, just paced
        one-per-global-round instead of one-per-class-round."""

        groups = defaultdict(list)
        for load in loads:
            groups[load["size_class"]].append(load)

        total = len(loads)
        selected = []
        used_buses = set()
        phase_kw_running = {"A": 0.0, "B": 0.0, "C": 0.0}

        class_state = {}
        for size_class, items in groups.items():
            quota = max(1, round(self.target_meters * len(items) / total))

            branches = defaultdict(list)
            for load in items:
                branches[load["branch"]].append(load)

            for branch in branches.values():
                branch.sort(key=lambda x: x["score"], reverse=True)

            class_state[size_class] = {
                "branch_queue": deque(branches.values()),
                "quota": quota,
                "count": 0,
            }

        active_classes = list(class_state.keys())

        while active_classes:
            progressed = False

            for size_class in list(active_classes):
                state = class_state[size_class]

                if state["count"] >= state["quota"]:
                    active_classes.remove(size_class)
                    continue

                load = self._pick_one_for_class(state, used_buses, phase_kw_running)

                if load is None:
                    active_classes.remove(size_class)
                    continue

                selected.append(load)
                used_buses.add(load["bus"])
                self._accumulate_phase_kw(phase_kw_running, load)
                state["count"] += 1
                progressed = True

                if state["count"] >= state["quota"]:
                    active_classes.remove(size_class)

            if not progressed:
                break

        if len(selected) > self.target_meters:
            # Truncate the incremental append order -- do NOT re-sort
            # by score. Re-sorting would discard both the round-robin
            # branch-diversity mechanism and the phase balancing just
            # applied above; only class-quota rounding can overshoot
            # here, by at most a few meters.
            selected = selected[:self.target_meters]

        return selected

    def _pick_one_for_class(self, state, used_buses, phase_kw_running):
        """Try each of this class's branches once, in rotation (the
        queue is rotated so the NEXT call for this class starts from a
        different branch -- preserving branch-diversity cycling across
        successive turns), returning the phase-aware best candidate
        from the first branch with one available. None if the class has
        nothing left to offer."""

        queue = state["branch_queue"]

        for _ in range(len(queue)):
            branch = queue[0]
            queue.rotate(-1)

            while branch and branch[0]["bus"] in used_buses:
                branch.pop(0)

            if not branch:
                continue

            load = self._pick_phase_aware(branch, phase_kw_running)
            branch.remove(load)
            return load

        return None

    # --------------------------------------------------
    # Stage 4 -- top-up (Step-3 defect #1), phase-aware
    # --------------------------------------------------

    def _top_up_shortfall(self, selected, all_loads):
        """Per-size-class quotas round independently and can undershoot
        target_meters (e.g. four classes each rounding down by a
        fraction). Fill the shortfall from the remaining candidates,
        respecting one-meter-per-bus, using the SAME incremental
        phase-aware pick as the main selection -- not a separate, score-
        only top-up followed by a separate balance pass."""

        shortfall = self.target_meters - len(selected)

        if shortfall <= 0:
            return selected

        used_buses = {load["bus"] for load in selected}
        phase_kw_running = self._selected_phase_kw(selected)

        candidates = sorted(
            (l for l in all_loads if l["bus"] not in used_buses),
            key=lambda x: x["score"],
            reverse=True,
        )

        for _ in range(shortfall):
            candidates = [l for l in candidates if l["bus"] not in used_buses]
            if not candidates:
                break

            load = self._pick_phase_aware(candidates, phase_kw_running)
            candidates.remove(load)

            selected.append(load)
            used_buses.add(load["bus"])
            self._accumulate_phase_kw(phase_kw_running, load)

        return selected

    # --------------------------------------------------
    # Incremental phase-balance criterion (sweep-defect #1 fix)
    # --------------------------------------------------

    def _phase_deficits(self, phase_kw_running):
        """Positive deficit[p] = phase p is under-represented, relative
        to its target share of PHASE_KW_BASELINE, given kW picked so
        far. All zero before the first pick (nothing to balance yet)."""

        total_baseline_kw = sum(self.PHASE_KW_BASELINE.values())
        target_share = {
            p: kw / total_baseline_kw for p, kw in self.PHASE_KW_BASELINE.items()
        }

        total_running = sum(phase_kw_running.values())
        if total_running == 0:
            return {p: 0.0 for p in self.PHASE_KW_BASELINE}

        return {
            p: target_share[p] - (phase_kw_running[p] / total_running)
            for p in self.PHASE_KW_BASELINE
        }

    def _pick_phase_aware(self, candidates, phase_kw_running):
        """candidates: sorted by score descending. Picks the highest-
        scoring candidate touching the currently most under-represented
        phase, if that phase has a positive deficit and such a
        candidate exists; otherwise falls back to the plain highest-
        scoring candidate (original behaviour). Deterministic: ties in
        the deficit or in score are broken by the fixed input order."""

        deficits = self._phase_deficits(phase_kw_running)
        neediest_phase = max(deficits, key=deficits.get)

        if deficits[neediest_phase] > 0:
            for candidate in candidates:
                if neediest_phase in self._load_phases(candidate):
                    return candidate

        return candidates[0]

    def _accumulate_phase_kw(self, phase_kw_running, load):
        for phase, kw in self._load_phase_kw(load).items():
            phase_kw_running[phase] += kw

    def _load_phases(self, load):
        """A load's phase field is 'A'/'B'/'C'/'AB'/'BC'/'CA'/'ABC' --
        return the list of single-letter phases it touches."""

        phase = load.get("phase", "")
        return [p for p in phase if p in ("A", "B", "C")] or ["A", "B", "C"]

    def _load_phase_kw(self, load):
        """A load's kW split evenly across the phases it touches (a
        line-to-line or 3-phase load contributes to more than one)."""

        phases = self._load_phases(load)
        share = load["base_kw"] / len(phases)
        return {p: share for p in phases}

    def _selected_phase_kw(self, selected):
        totals = {"A": 0.0, "B": 0.0, "C": 0.0}
        for load in selected:
            for phase, kw in self._load_phase_kw(load).items():
                totals[phase] += kw
        return totals

def _one_load_per_bus(loads):
    """Where a bus hosts more than one Load object (S49a/b/c, S65a/b/c,
    S76a/b/c -- per-phase splits of one 3-phase spot load, per model
    verification Q2), keep the highest-kW one as that bus's
    representative. All placement strategies enforce one meter per bus;
    Hybrid does this implicitly via used_buses tracking in its
    round-robin. Random and Degree need the same collapse done
    explicitly before choosing among buses."""

    by_bus = {}
    for load in loads:
        bus = load["bus"]
        if bus not in by_bus or load["base_kw"] > by_bus[bus]["base_kw"]:
            by_bus[bus] = load
    return list(by_bus.values())


class RandomPlacement(PlacementStrategy):
    """Uninformed baseline: uniform-random selection of target_meters
    distinct buses, one load per bus. Seeded for reproducibility;
    instantiate with a different seed per draw to build the between-seed
    variance distribution RES.md D10 Phase 4 sub-study 4A asks for at low
    meter counts -- this class deliberately does not average across
    seeds itself, so that variance is visible to the caller."""

    METER_CEILING = 85

    def __init__(self, target_meters=REFERENCE_TARGET_METERS, seed=0):
        if target_meters > self.METER_CEILING:
            raise ValueError(
                f"target_meters={target_meters} exceeds the meter ceiling "
                f"({self.METER_CEILING} distinct load buses)."
            )
        self.target_meters = target_meters
        self.seed = seed

    def select(self, loads, context):
        candidates = _one_load_per_bus(loads)

        if len(candidates) < self.target_meters:
            raise ValueError(
                f"only {len(candidates)} distinct load buses available, "
                f"cannot select {self.target_meters}"
            )

        rng = random.Random(self.seed)
        selected = rng.sample(candidates, self.target_meters)

        assert len(selected) == self.target_meters
        assert len({l["bus"] for l in selected}) == len(selected)

        return selected


class DegreePlacement(PlacementStrategy):
    """Topology-only baseline: the target_meters distinct buses with the
    highest topology-graph degree (branch points first), one load per
    bus. Ties broken by higher kW, then bus id, for determinism."""

    METER_CEILING = 85

    def __init__(self, target_meters=REFERENCE_TARGET_METERS):
        if target_meters > self.METER_CEILING:
            raise ValueError(
                f"target_meters={target_meters} exceeds the meter ceiling "
                f"({self.METER_CEILING} distinct load buses)."
            )
        self.target_meters = target_meters

    def select(self, loads, context):
        candidates = _one_load_per_bus(loads)

        if len(candidates) < self.target_meters:
            raise ValueError(
                f"only {len(candidates)} distinct load buses available, "
                f"cannot select {self.target_meters}"
            )

        ranked = sorted(
            candidates,
            key=lambda l: (
                -context.bus_degree(l["bus"]),
                -l["base_kw"],
                str(l["bus"]),
            ),
        )

        selected = ranked[:self.target_meters]

        assert len(selected) == self.target_meters
        assert len({l["bus"] for l in selected}) == len(selected)

        return selected


class CommunityDetectionPlacement(PlacementStrategy):
    """RES.md D10 Phase 4 sub-study 4A's strong baseline, per XuIoT26
    (Xu et al., "A Multiarea Data Reconstruction Framework to Mitigate
    FDIA in IoT-Enabled Power Distribution Systems," IEEE IoT J.
    13(14):31737-31751, 2026), Section II-B.

    XuIoT26'S, verified/reused as-is:
      - equivalent electrical distance d_ij, eq. (4)-(6)
        (placement/electrical_distance.py) -- verified against the
        paper's own worked examples (eq. 33-38) to within rounding;
        see tests/test_electrical_distance.py.

    XuIoT26'S, with documented simplifications (placement/community_detection.py):
      - eq. (7)-(9)'s edge-graph transformation is NOT implemented --
        eq. (9)'s omega_ij term is undefined anywhere in the paper, and
        eq. (10) is written over node-graph quantities (A_ij,
        kappa_i*kappa_j) despite the prose calling the edge graph F the
        adjacency matrix being maximised over. Standard weighted
        Louvain modularity maximisation runs directly on the bus graph
        instead, with A_ij = d_ij. The paper states the edge-graph
        step's purpose is "scale equalization of each subregion" -- a
        balancing refinement, not the core partitioning mechanism --
        so this is a real, documented simplification, not a silent one.
      - The prose claims smaller d_ij means tighter coupling and a
        larger edge weight; the paper's OWN arithmetic says the
        opposite (d_17=4.09, three-phase/tight, vs d_12=1.31,
        single-phase/loose -- larger d for the tighter connection).
        Tested empirically, not assumed: weight=d_ij produces
        electrically sensible, lateral-following communities
        (modularity 0.951, ~10 substantive communities sized 3-24,
        2/3 of the paper's named boundary nodes reproduced); weight=
        1/d_ij produces one 53-node community absorbing most of the
        feeder plus fragments (modularity 0.74) -- not sensible.
        weight=d_ij is used. Full comparison in
        reports/community_detection_report.txt.

    OURS (no source -- XuIoT26 partitions for state-estimation data
    reconstruction, not meter placement):
      - Stage 3, meter budget allocation across communities
        proportional to each community's share of connected kW
        (largest-remainder / Hare-Niemeyer apportionment, exact),
        highest-kW load selected first within each community, with a
        global top-up if a community's allocation exceeds its
        available distinct load buses.
      - The MERGED variant (num_communities=4): agglomeratively merges
        the raw 12 Louvain communities down to a target count by
        strongest inter-community coupling (community_detection.
        merge_communities()). This is our own adaptation to approximate
        the effect of the un-implementable eq.(7)-(9) scale-
        equalisation step, chosen to match XuIoT26's reported subarea
        count -- not a re-derivation of that step. BOTH the raw and
        merged variants are kept selectable (num_communities=None vs.
        an int): the raw result is the faithful output of the equations
        we CAN implement, and neither supersedes the other. See
        reports/community_detection_report.txt.

    Requires the circuit to be compiled and solved (SystemY needs a
    live OpenDSS session) -- call after core.ieee123_model.extract_all().
    """

    METER_CEILING = 85
    LOUVAIN_RESOLUTION = 1.0  # standard; see community_detection.py docstring

    # Class-level cache: the raw Louvain result depends only on the
    # feeder's fixed topology, not on target_meters or the merged
    # variant, so it's computed once and shared by every instance in
    # this process (a Phase-4-style sweep instantiates this class many
    # times per run -- recomputing SystemY + Louvain per instantiation
    # would dominate runtime). clear_cache() resets it explicitly.
    _raw_cache = None  # (community_of, adjacency)

    def __init__(self, target_meters=REFERENCE_TARGET_METERS, num_communities=None):
        if target_meters > self.METER_CEILING:
            raise ValueError(
                f"target_meters={target_meters} exceeds the meter ceiling "
                f"({self.METER_CEILING} distinct load buses)."
            )
        self.target_meters = target_meters
        self.num_communities = num_communities  # None = raw Louvain result
        self._community_of = None  # resolved (raw or merged) variant, cached per instance

    @classmethod
    def clear_cache(cls):
        cls._raw_cache = None

    @classmethod
    def _get_raw_communities(cls, context):
        if cls._raw_cache is not None:
            return cls._raw_cache

        from placement.electrical_distance import (
            get_system_y_index,
            build_electrical_distance_graph,
        )
        from placement.community_detection import (
            louvain_communities,
            relabel_communities,
        )

        Y, node_index = get_system_y_index()
        adjacency = build_electrical_distance_graph(context, Y, node_index)
        adjacency = {
            i: {j: float(w) for j, w in nbrs.items()}
            for i, nbrs in adjacency.items()
        }

        community_of = relabel_communities(
            louvain_communities(adjacency, resolution=cls.LOUVAIN_RESOLUTION)
        )

        cls._raw_cache = (community_of, adjacency)
        return cls._raw_cache

    def _compute_communities(self, context):
        if self._community_of is not None:
            return self._community_of

        from placement.community_detection import merge_communities, relabel_communities

        raw_community_of, adjacency = self._get_raw_communities(context)

        if self.num_communities is None:
            self._community_of = raw_community_of
        else:
            merged = merge_communities(adjacency, raw_community_of, self.num_communities)
            self._community_of = relabel_communities(merged)

        return self._community_of

    def select(self, loads, context):
        community_of = self._compute_communities(context)

        candidates = _one_load_per_bus(loads)

        if len(candidates) < self.target_meters:
            raise ValueError(
                f"only {len(candidates)} distinct load buses available, "
                f"cannot select {self.target_meters}"
            )

        by_community = defaultdict(list)
        for load in candidates:
            c = community_of.get(str(load["bus"]))
            by_community[c].append(load)

        for items in by_community.values():
            items.sort(key=lambda l: (-l["base_kw"], str(l["bus"])))

        community_kw = {
            c: sum(l["base_kw"] for l in items) for c, items in by_community.items()
        }

        allocation = self._allocate_budget_by_kw(community_kw, self.target_meters)

        capped = {
            c: min(alloc, len(by_community[c])) for c, alloc in allocation.items()
        }

        # ROUND-ROBIN across communities (RES.md D10 Phase 4 sub-study
        # 4A strategy fix), not block concatenation. Rationale: NESTED
        # construction takes an ordered prefix of a full-count run --
        # block concatenation ("all of community 0, then all of
        # community 1, ...") means a short prefix is dominated by
        # whichever one or two communities sort first, leaving most
        # communities entirely unrepresented even when their final
        # allocation is nonzero. Verified empirically before this fix:
        # at the 85-meter run's first-9 prefix, only 2 of 9 raw Louvain
        # communities appeared at all. Interleaving one pick per
        # community per round (mirroring HybridPlacement's own
        # round-robin branch queue) means every community with a
        # nonzero allocation is represented within the first
        # (num_communities_with_nonzero_allocation) picks of ANY
        # nested prefix, not just the final target_meters set.
        community_order = sorted(capped, key=str)
        queues = {c: list(by_community[c][: capped[c]]) for c in community_order}
        selected = []
        progressed = True
        while progressed:
            progressed = False
            for c in community_order:
                if queues[c]:
                    selected.append(queues[c].pop(0))
                    progressed = True

        shortfall = self.target_meters - len(selected)

        if shortfall > 0:
            used_buses = {l["bus"] for l in selected}
            remaining = sorted(
                (l for l in candidates if l["bus"] not in used_buses),
                key=lambda l: (-l["base_kw"], str(l["bus"])),
            )
            selected.extend(remaining[:shortfall])

        assert len(selected) == self.target_meters, (
            f"selected {len(selected)} meters, expected {self.target_meters}"
        )
        assert len({l["bus"] for l in selected}) == len(selected), (
            "duplicate bus in selection"
        )

        return selected

    @staticmethod
    def _allocate_budget_by_kw(community_kw, target_meters):
        """Largest-remainder (Hare-Niemeyer) apportionment: proportional
        to each community's share of connected kW, exact sum ==
        target_meters, deterministic tie-break by community id. Then a
        MINIMUM-SHARE top-up (RES.md D10 Phase 4 sub-study 4A strategy
        fix): pure largest-remainder can leave several communities at
        exactly 0 when target_meters is small relative to the number of
        communities -- e.g. 2 of 9 raw Louvain communities at
        target_meters=9, confirmed empirically. That is mathematically
        correct apportionment, but it means this strategy's SET, not
        just its nested-prefix ORDER (fixed separately in select()
        above), stops being electrically diverse exactly in the
        low-density regime this project studies -- the strong baseline
        must not be weakened by an implementation choice here. Once
        every community has been floored to >=1 (feasible iff
        target_meters >= number of communities, guaranteed by
        conservation of the fixed total), donate one meter to each
        zero-share community from the CURRENTLY largest allocation,
        largest first -- the same logic apportionment methods use to
        guarantee every represented group a seat. When target_meters <
        number of communities, a floor of 1 for every community is not
        achievable at all; the plain largest-remainder result is
        returned unmodified in that case (stated here, not silent)."""

        total_kw = sum(community_kw.values())

        if total_kw == 0 or not community_kw:
            return {c: 0 for c in community_kw}

        raw = {c: target_meters * kw / total_kw for c, kw in community_kw.items()}
        floor_alloc = {c: int(v) for c, v in raw.items()}

        remainder = target_meters - sum(floor_alloc.values())

        by_fraction = sorted(
            raw.items(),
            key=lambda item: (-(item[1] - floor_alloc[item[0]]), str(item[0])),
        )

        for c, _ in by_fraction[:remainder]:
            floor_alloc[c] += 1

        if target_meters < len(community_kw):
            return floor_alloc  # a floor of 1 for every community is infeasible

        zero_share = [c for c, v in floor_alloc.items() if v == 0]
        for c in sorted(zero_share, key=str):
            donor = max(floor_alloc, key=lambda k: (floor_alloc[k], str(k)))
            floor_alloc[donor] -= 1
            floor_alloc[c] += 1

        return floor_alloc
