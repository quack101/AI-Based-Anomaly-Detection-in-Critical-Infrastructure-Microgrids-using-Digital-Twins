"""
HybridV2Placement -- HybridPlacement with two changes to HOW the next
branch is chosen. Everything else (one load per bus, size-class quotas,
scoring weights, within-branch score order, incremental phase-aware
pick, top-up) is inherited unchanged. HybridPlacement itself is not
modified, so v1 and v2 can be compared side by side.

Why (experiments/compare_placements.py): v1 visits each size class's
branches in fixed bus-number order. At small budgets it exhausts its
quota before reaching the high-numbered side of the feeder -- at 30
meters every v1 meter sits on buses 1-53 and the 39 load buses on
64-114 get none (avg 5.0 hops to the nearest meter vs 1.6 for random).

Change 1 -- farthest-first branch order. On each class's turn, among
its branches that still have an available load, pick by:
  (a) fewest meters already in that branch (feeder-wide), so every
      branch still gets one meter before any gets a second -- this is
      v1's branch-diversity mechanism, kept;
  (b) then the largest hop distance from that branch's candidate
      buses to the nearest already-selected meter (k-center-style
      spread across the feeder graph);
  (c) then v1's original queue order, for determinism.

Change 2 -- phase look-ahead. If a phase is under-represented and the
top-ranked branch has no candidate on that phase, the next
PHASE_LOOKAHEAD branches in the SAME fewest-meters tier are checked and
the first one offering that phase is taken instead. Restricted to the
same tier so phase balancing never costs branch diversity.

The within-branch pick is still v1's _pick_phase_aware (best score,
preferring the neediest phase), so the weights keep the role they had.
"""

from collections import deque

from placement.placement_strategy import HybridPlacement, REFERENCE_TARGET_METERS


class HybridV2Placement(HybridPlacement):
    PHASE_LOOKAHEAD = 3

    def __init__(self, target_meters=REFERENCE_TARGET_METERS, verbose=False, weights=None,
                 phase_lookahead=PHASE_LOOKAHEAD):
        super().__init__(target_meters=target_meters, verbose=verbose, weights=weights)
        self.phase_lookahead = phase_lookahead

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def select(self, loads, context):
        self._graph = context.topology_graph
        self._dist_cache = {}
        self._branch_meters = {}
        return super().select(loads, context)

    # --------------------------------------------------
    # Distances on the feeder graph
    # --------------------------------------------------

    def _hops_from(self, bus):
        """BFS hop distance from one bus to every bus, cached."""
        bus = str(bus)
        if bus not in self._dist_cache:
            dist = {bus: 0}
            q = deque([bus])
            while q:
                u = q.popleft()
                for v in self._graph.get(u, []):
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        q.append(v)
            self._dist_cache[bus] = dist
        return self._dist_cache[bus]

    def _distance_to_selected(self, bus, used_buses):
        if not used_buses:
            return float("inf")
        dist = self._hops_from(bus)
        return min(dist.get(str(s), float("inf")) for s in used_buses)

    # --------------------------------------------------
    # Branch choice (overrides v1's fixed rotation)
    # --------------------------------------------------

    def _pick_one_for_class(self, state, used_buses, phase_kw_running):
        queue = state["branch_queue"]

        candidates = []
        for order, branch in enumerate(queue):
            while branch and branch[0]["bus"] in used_buses:
                branch.pop(0)
            available = [l for l in branch if l["bus"] not in used_buses]
            if not available:
                continue
            branch_id = available[0]["branch"]
            spread = max(self._distance_to_selected(l["bus"], used_buses) for l in available)
            candidates.append({
                "branch": branch, "available": available,
                "key": (self._branch_meters.get(branch_id, 0), -spread, order),
            })

        if not candidates:
            return None

        candidates.sort(key=lambda c: c["key"])
        chosen = candidates[0]

        deficits = self._phase_deficits(phase_kw_running)
        neediest = max(deficits, key=deficits.get)
        if deficits[neediest] > 0 and not self._offers_phase(chosen["available"], neediest):
            tier = chosen["key"][0]
            for c in candidates[1:1 + self.phase_lookahead]:
                if c["key"][0] != tier:
                    break
                if self._offers_phase(c["available"], neediest):
                    chosen = c
                    break

        load = self._pick_phase_aware(chosen["available"], phase_kw_running)
        chosen["branch"].remove(load)
        self._branch_meters[load["branch"]] = self._branch_meters.get(load["branch"], 0) + 1
        return load

    def _offers_phase(self, loads, phase):
        return any(phase in self._load_phases(l) for l in loads)
