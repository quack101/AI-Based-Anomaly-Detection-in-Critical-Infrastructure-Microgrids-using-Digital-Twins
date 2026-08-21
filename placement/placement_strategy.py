"""
Meter-placement strategies for RES.md D10 Phase 4 sub-study 4A (meter-
density sweep x placement strategy). All four strategies share one
FeederContext (feeder_context.py) computed once from (network, loads).

HybridPlacement is the original HybridRepresentativePlacement, moved
here and defect-fixed (see class docstring). RandomPlacement,
DegreePlacement and CommunityDetectionPlacement (stub) are new.
"""

from abc import ABC, abstractmethod
from collections import defaultdict
import math
import random


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
    phase balancing.
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

    def __init__(self, target_meters=49, verbose=False):
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

        selected = self._balance_phases(selected, loads)

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
    # Stage 3 -- quota-based round-robin selection
    # --------------------------------------------------

    def _representative_selection(self, loads):
        groups = defaultdict(list)

        for load in loads:
            groups[load["size_class"]].append(load)

        total = len(loads)
        selected = []

        for size_class, items in groups.items():
            quota = max(1, round(self.target_meters * len(items) / total))

            branches = defaultdict(list)
            for load in items:
                branches[load["branch"]].append(load)

            for branch in branches.values():
                branch.sort(key=lambda x: x["score"], reverse=True)

            used_buses = {load["bus"] for load in selected}
            count = 0

            while count < quota:
                added = False

                for branch in branches.values():
                    while branch and branch[0]["bus"] in used_buses:
                        branch.pop(0)

                    if not branch:
                        continue

                    load = branch.pop(0)
                    selected.append(load)
                    used_buses.add(load["bus"])

                    count += 1
                    added = True

                    if count == quota:
                        break

                if not added:
                    break

        if len(selected) > self.target_meters:
            selected.sort(key=lambda x: x["score"], reverse=True)
            selected = selected[:self.target_meters]

        return selected

    # --------------------------------------------------
    # Stage 4 -- top-up (Step-3 defect #1)
    # --------------------------------------------------

    def _top_up_shortfall(self, selected, all_loads):
        """Per-size-class quotas round independently and can undershoot
        target_meters (e.g. four classes each rounding down by a
        fraction). Fill the shortfall from the highest-scoring
        remaining loads, respecting one-meter-per-bus, without touching
        the quota/round-robin logic above."""

        shortfall = self.target_meters - len(selected)

        if shortfall <= 0:
            return selected

        used_buses = {load["bus"] for load in selected}

        candidates = sorted(
            (l for l in all_loads if l["bus"] not in used_buses),
            key=lambda x: x["score"],
            reverse=True,
        )

        for load in candidates:
            if shortfall == 0:
                break

            if load["bus"] in used_buses:
                continue

            selected.append(load)
            used_buses.add(load["bus"])
            shortfall -= 1

        return selected

    # --------------------------------------------------
    # Stage 5 -- phase balancing (Step-3 defect #2)
    # --------------------------------------------------

    def _balance_phases(self, selected, all_loads):
        """Post-selection swap pass: for each over-represented phase,
        exchange the lowest-scoring selection on that phase for the
        highest-scoring unselected load on the most under-represented
        phase, balanced against connected kW per phase
        (PHASE_KW_BASELINE), not node count. Stops when no swap reduces
        the maximum phase-kW deviation from the target share, or when
        no eligible swap candidate remains."""

        total_baseline_kw = sum(self.PHASE_KW_BASELINE.values())
        target_share = {
            phase: kw / total_baseline_kw
            for phase, kw in self.PHASE_KW_BASELINE.items()
        }

        used_buses = {l["bus"] for l in selected}
        # One candidate per bus -- otherwise a sibling load on an
        # already-selected bus (e.g. S49b, when S49a is selected) can
        # be swapped in later and produce a duplicate bus.
        unselected = _one_load_per_bus(
            [l for l in all_loads if l["bus"] not in used_buses]
        )

        max_swaps = len(selected)  # hard bound against pathological loops

        for _ in range(max_swaps):
            phase_kw = self._selected_phase_kw(selected)
            total_kw = sum(phase_kw.values()) or 1.0
            deviation = {
                phase: (phase_kw[phase] / total_kw) - target_share[phase]
                for phase in target_share
            }

            over_phase = max(deviation, key=deviation.get)
            under_phase = min(deviation, key=deviation.get)

            if deviation[over_phase] <= 0:
                break  # nothing over-represented enough to fix

            worst_in_over = self._lowest_scoring_on_phase(selected, over_phase)
            best_candidate = self._highest_scoring_on_phase(unselected, under_phase)

            if worst_in_over is None or best_candidate is None:
                break

            projected = dict(phase_kw)
            projected[over_phase] -= self._load_phase_kw(worst_in_over).get(over_phase, 0.0)
            projected[under_phase] += self._load_phase_kw(best_candidate).get(under_phase, 0.0)
            projected_total = sum(projected.values()) or 1.0
            projected_deviation = max(
                abs((projected[p] / projected_total) - target_share[p])
                for p in target_share
            )
            current_deviation = max(abs(d) for d in deviation.values())

            if projected_deviation >= current_deviation:
                break  # swap would not improve balance -- stop

            selected.remove(worst_in_over)
            unselected.remove(best_candidate)
            selected.append(best_candidate)
            unselected.append(worst_in_over)
            used_buses.discard(worst_in_over["bus"])
            used_buses.add(best_candidate["bus"])

        return selected

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

    def _lowest_scoring_on_phase(self, selected, phase):
        on_phase = [l for l in selected if phase in self._load_phases(l)]
        if not on_phase:
            return None
        return min(on_phase, key=lambda l: l["score"])

    def _highest_scoring_on_phase(self, candidates, phase):
        on_phase = [l for l in candidates if phase in self._load_phases(l)]
        if not on_phase:
            return None
        return max(on_phase, key=lambda l: l["score"])


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

    def __init__(self, target_meters=49, seed=0):
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

    def __init__(self, target_meters=49):
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
    """Stub. RES.md D10 Phase 4 sub-study 4A calls for a community-
    detection placement strategy citing XuIoT26 eq. (1)-(10) (graph
    community partitioning to spread meters across electrically
    distinct communities rather than by score or degree alone). Not
    implemented in this task -- raises NotImplementedError rather than
    silently returning a placeholder result, per CLAUDE.md's "no fake
    outputs" rule."""

    def __init__(self, target_meters=49):
        self.target_meters = target_meters

    def select(self, loads, context):
        raise NotImplementedError(
            "CommunityDetectionPlacement is a stub (RES.md D10 Phase 4 "
            "sub-study 4A, citing XuIoT26 eq. 1-10). Not implemented."
        )
