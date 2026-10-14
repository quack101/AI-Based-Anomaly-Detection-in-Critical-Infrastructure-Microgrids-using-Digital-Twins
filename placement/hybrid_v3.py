"""
HybridV3Placement -- HybridV2Placement that prefers well-connected buses.

Why (validation/literature_metrics.py boi, 49 meters): Degree placement
has the highest redundancy (SORI 766 vs 664 set A, 576 set B) because it
never meters a dead-end bus. Of the 85 load buses, 36 are dead ends
(topology degree 1). A meter on a dead end is seen by few measurement
rows and sees only one neighbour, so it adds little redundancy. v2's
farthest-first branch order makes this worse (set B: 32 dead ends),
because the farthest bus is usually the end of a line. Degree, however,
ignores phase balance (6.4 pp, over the 5 pp tolerance of
tests/test_placement_regression.py) and size classes.

Change (on top of v2): a load is "connected" if its bus has degree >= 2.
  1. Within a branch, only connected loads are offered when the branch
     has any; dead ends are the fallback.
  2. Branch order key gains "branch has no connected load", placed by
     connection_priority:
       "within_tier": (meters in branch, no connected load,
           -spread, order) -- v2's branch diversity first, so a branch
           with only dead ends still gets its meter before any branch
           gets a second one;
       "first": (no connected load, meters in branch, -spread, order) --
           every connected load anywhere before any dead end, branch
           diversity second;
       "first_balanced" (default): as "first", but when no branch in the top tier
           can serve the most under-represented phase, take the
           first-ranked branch with a connected load on that phase, else
           the first-ranked load on that phase even if it is a dead end
           (phase balance before connectivity).
     The phase look-ahead itself stays within the same tier (first two
     keys).
Size-class quotas, scores, phase-aware pick and top-up are inherited.
"""

from placement.hybrid_v2 import HybridV2Placement
from placement.placement_strategy import REFERENCE_TARGET_METERS


class HybridV3Placement(HybridV2Placement):
    CONNECTION_PRIORITIES = ("within_tier", "first", "first_balanced")

    def __init__(self, target_meters=REFERENCE_TARGET_METERS, verbose=False, weights=None,
                 phase_lookahead=HybridV2Placement.PHASE_LOOKAHEAD, connection_priority="first_balanced"):
        super().__init__(target_meters=target_meters, verbose=verbose, weights=weights,
                         phase_lookahead=phase_lookahead)
        if connection_priority not in self.CONNECTION_PRIORITIES:
            raise ValueError(f"connection_priority must be one of {self.CONNECTION_PRIORITIES}")
        self.connection_priority = connection_priority

    def select(self, loads, context):
        self._context = context
        return super().select(loads, context)

    def _is_connected(self, load):
        return self._context.bus_degree(load["bus"]) >= 2

    def _pick_one_for_class(self, state, used_buses, phase_kw_running):
        queue = state["branch_queue"]

        candidates = []
        for order, branch in enumerate(queue):
            available = [l for l in branch if l["bus"] not in used_buses]
            if not available:
                continue
            connected = [l for l in available if self._is_connected(l)]
            pool = connected or available
            spread = max(self._distance_to_selected(l["bus"], used_buses) for l in pool)
            meters = self._branch_meters.get(available[0]["branch"], 0)
            dead_end_only = not connected
            key = ((meters, dead_end_only) if self.connection_priority == "within_tier"
                   else (dead_end_only, meters)) + (-spread, order)
            candidates.append({"branch": branch, "pool": pool, "available": available, "key": key})

        if not candidates:
            return None

        candidates.sort(key=lambda c: c["key"])
        chosen = candidates[0]

        deficits = self._phase_deficits(phase_kw_running)
        neediest = max(deficits, key=deficits.get)
        if deficits[neediest] > 0 and not self._offers_phase(chosen["pool"], neediest):
            tier = chosen["key"][:2]
            found = False
            for c in candidates[1:1 + self.phase_lookahead]:
                if c["key"][:2] != tier:
                    break
                if self._offers_phase(c["pool"], neediest):
                    chosen, found = c, True
                    break
            if not found and self.connection_priority == "first_balanced":
                # any connected load on the needed phase, then any dead end on it
                for pool_name in ("pool", "available"):
                    c = next((c for c in candidates if self._offers_phase(c[pool_name], neediest)), None)
                    if c is not None:
                        chosen = dict(c, pool=c[pool_name])
                        break

        load = self._pick_phase_aware(chosen["pool"], phase_kw_running)
        chosen["branch"].remove(load)
        self._branch_meters[load["branch"]] = self._branch_meters.get(load["branch"], 0) + 1
        return load
