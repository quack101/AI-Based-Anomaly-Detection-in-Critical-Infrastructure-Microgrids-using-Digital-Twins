"""
READ-ONLY scratch script (not committed) -- compares HybridPlacement
(v1), HybridV2Placement (v2), and RandomPlacement (20-seed average) at
target_meters=49, entirely in memory. Does not call
save_representative_meters, does not run select_meters.py or
generate_registry.py, does not touch simulation/representative_meters.json
or config/sensor_registry*.json.

Run: python reports/_scratch_placement_v1_v2_compare.py
"""

import json
import statistics as stats
from collections import deque
from pathlib import Path

from core.ieee123_model import extract_all
from placement.feeder_context import FeederContext
from placement.feeder_adapter import network_and_loads_from_core_extraction
from placement.placement_strategy import HybridPlacement, RandomPlacement
from placement.hybrid_v2 import HybridV2Placement

BASE_DIR = Path(__file__).resolve().parent.parent
SAVED_METERS_PATH = BASE_DIR / "simulation" / "representative_meters.json"
TARGET = 49
N_RANDOM_SEEDS = 20


def load_phases(load):
    phase = load.get("phase", "")
    return [p for p in phase if p in ("A", "B", "C")] or ["A", "B", "C"]


def bfs_hops(graph, sources):
    """Multi-source BFS hop distance from any bus in `sources` to every
    reachable bus in `graph`."""
    dist = {s: 0 for s in sources}
    q = deque(sources)
    while q:
        u = q.popleft()
        for v in graph.get(u, []):
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def hop_stats(topology_graph, metered_buses):
    """Bus '610' has no edges at all in context.topology_graph (its only
    link is via an open tie switch, which build_topology_graph() drops
    entirely) -- it is structurally isolated for EVERY strategy, not a
    placement-quality signal. Excluded from mean/p95/max/n_4plus, but
    counted separately."""
    dist = bfs_hops(topology_graph, [str(b) for b in metered_buses])
    all_buses = list(topology_graph.keys())
    reachable_buses = [b for b in all_buses if b in dist]
    isolated_buses = [b for b in all_buses if b not in dist]
    hops = [dist[b] for b in reachable_buses]
    n_4plus = sum(1 for h in hops if h >= 4)
    return {
        "n_buses": len(all_buses),
        "n_isolated": len(isolated_buses),
        "isolated_buses": isolated_buses,
        "mean": stats.mean(hops),
        "p95": sorted(hops)[int(round(0.95 * (len(hops) - 1)))],
        "max": max(hops),
        "n_4plus_hops": n_4plus,
        "dist": dist,
    }


def kw_share(selected, all_loads):
    metered_buses = {str(l["bus"]) for l in selected}
    metered_kw = sum(l["base_kw"] for l in all_loads if str(l["bus"]) in metered_buses)
    total_kw = sum(l["base_kw"] for l in all_loads)
    return metered_kw / total_kw if total_kw else 0.0


def phase_counts(selected):
    counts = {"A": 0, "B": 0, "C": 0}
    for l in selected:
        for p in load_phases(l):
            counts[p] += 1
    return counts


def bus_range_distance(dist, lo, hi):
    """For bus ids that parse as int in [lo, hi]: list of (bus, hop)."""
    out = []
    for bus, hop in dist.items():
        try:
            n = int(bus)
        except ValueError:
            continue
        if lo <= n <= hi:
            out.append((bus, hop))
    return sorted(out, key=lambda x: int(x[0]))


def main():
    extraction = extract_all()
    network, loads = network_and_loads_from_core_extraction(extraction)
    context = FeederContext(network, loads)
    context.annotate_loads(loads)

    with open(SAVED_METERS_PATH) as f:
        saved_meters = json.load(f)
    saved_buses = sorted({str(m["bus"]) for m in saved_meters}, key=lambda b: (len(b), b))
    saved_load_names = sorted(m["load_name"] for m in saved_meters)

    # ---- v1, freshly run in memory ----
    v1_strategy = HybridPlacement(target_meters=TARGET)
    v1_selected = v1_strategy.select([dict(l) for l in loads], context)
    v1_buses = sorted({str(l["bus"]) for l in v1_selected}, key=lambda b: (len(b), b))
    v1_load_names = sorted(l["name"] for l in v1_selected)

    identical_to_saved = (v1_load_names == saved_load_names)

    # ---- v2 ----
    v2_strategy = HybridV2Placement(target_meters=TARGET)
    v2_selected = v2_strategy.select([dict(l) for l in loads], context)
    v2_buses = sorted({str(l["bus"]) for l in v2_selected}, key=lambda b: (len(b), b))

    # ---- random, 20 seeds, averaged ----
    random_hop_stats = []
    random_kw_shares = []
    random_phase_counts = []
    random_overlap_v1 = []
    for seed in range(N_RANDOM_SEEDS):
        r_strategy = RandomPlacement(target_meters=TARGET, seed=seed)
        r_selected = r_strategy.select([dict(l) for l in loads], context)
        r_buses = {str(l["bus"]) for l in r_selected}
        hs = hop_stats(context.topology_graph, r_buses)
        random_hop_stats.append(hs)
        random_kw_shares.append(kw_share(r_selected, loads))
        random_phase_counts.append(phase_counts(r_selected))
        random_overlap_v1.append(len(r_buses & set(v1_buses)))

    def avg(key):
        return stats.mean(h[key] for h in random_hop_stats)

    rnd_avg = {
        "mean": avg("mean"), "p95": avg("p95"), "max": avg("max"),
        "n_4plus_hops": avg("n_4plus_hops"),
    }
    rnd_kw_share_avg = stats.mean(random_kw_shares)
    rnd_phaseA = stats.mean(p["A"] for p in random_phase_counts)
    rnd_phaseB = stats.mean(p["B"] for p in random_phase_counts)
    rnd_phaseC = stats.mean(p["C"] for p in random_phase_counts)
    rnd_overlap_v1_avg = stats.mean(random_overlap_v1)

    v1_hs = hop_stats(context.topology_graph, v1_buses)
    v2_hs = hop_stats(context.topology_graph, v2_buses)

    overlap_buses = sorted(set(v1_buses) & set(v2_buses), key=lambda b: (len(b), b))

    v1_kw_share = kw_share(v1_selected, loads)
    v2_kw_share = kw_share(v2_selected, loads)

    v1_phase = phase_counts(v1_selected)
    v2_phase = phase_counts(v2_selected)

    v1_8083 = bus_range_distance(v1_hs["dist"], 80, 83)
    v2_8083 = bus_range_distance(v2_hs["dist"], 80, 83)
    v1_64114 = bus_range_distance(v1_hs["dist"], 64, 114)
    v2_64114 = bus_range_distance(v2_hs["dist"], 64, 114)

    print("=== IDENTICAL TO SAVED FILE ===")
    print("v1 fresh-run load names == saved representative_meters.json load names:", identical_to_saved)
    if not identical_to_saved:
        only_saved = sorted(set(saved_load_names) - set(v1_load_names))
        only_v1 = sorted(set(v1_load_names) - set(saved_load_names))
        print("only in saved:", only_saved)
        print("only in fresh v1:", only_v1)
    print()

    print("=== COUNTS ===")
    print("v1 n_meters:", len(v1_selected), "v2 n_meters:", len(v2_selected))
    print()

    print("=== OVERLAP v1/v2 ===")
    print("overlap count:", len(overlap_buses))
    print("overlap buses:", overlap_buses)
    print("avg random/v1 overlap (20 seeds):", rnd_overlap_v1_avg)
    print()

    print("=== HOP STATS (to nearest meter, over", v1_hs["n_buses"] - v1_hs["n_isolated"], "reachable topology buses;",
          v1_hs["n_isolated"], "structurally isolated bus(es) excluded:", v1_hs["isolated_buses"], ") ===")
    print(f"v1:     mean={v1_hs['mean']:.3f} p95={v1_hs['p95']} max={v1_hs['max']} n_4plus={v1_hs['n_4plus_hops']}")
    print(f"v2:     mean={v2_hs['mean']:.3f} p95={v2_hs['p95']} max={v2_hs['max']} n_4plus={v2_hs['n_4plus_hops']}")
    print(f"random: mean={rnd_avg['mean']:.3f} p95={rnd_avg['p95']:.3f} max={rnd_avg['max']:.3f} n_4plus={rnd_avg['n_4plus_hops']:.3f}")
    print()

    print("=== KW SHARE ON METERED BUS ===")
    print(f"v1: {v1_kw_share:.4f}  v2: {v2_kw_share:.4f}  random(avg): {rnd_kw_share_avg:.4f}")
    print()

    print("=== METERS PER PHASE ===")
    print("v1:", v1_phase, " v2:", v2_phase, f" random(avg): A={rnd_phaseA:.2f} B={rnd_phaseB:.2f} C={rnd_phaseC:.2f}")
    print()

    print("=== BUSES 80-83 ===")
    print("v1:", v1_8083)
    print("v2:", v2_8083)
    print()

    print("=== BUSES 64-114 ===")
    print("v1:", v1_64114)
    print("v2:", v2_64114)
    n_v1_64114_covered = sum(1 for _, h in v1_64114 if h == 0)
    n_v2_64114_covered = sum(1 for _, h in v2_64114 if h == 0)
    print(f"v1 buses in 64-114 directly metered (hop=0): {n_v1_64114_covered}/{len(v1_64114)}")
    print(f"v2 buses in 64-114 directly metered (hop=0): {n_v2_64114_covered}/{len(v2_64114)}")
    v1_64114_hops = [h for _, h in v1_64114]
    v2_64114_hops = [h for _, h in v2_64114]
    print(f"v1 64-114 hop mean/max: {stats.mean(v1_64114_hops):.3f}/{max(v1_64114_hops)}")
    print(f"v2 64-114 hop mean/max: {stats.mean(v2_64114_hops):.3f}/{max(v2_64114_hops)}")


if __name__ == "__main__":
    main()
