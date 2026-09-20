"""
Stage 2 (Louvain community detection) and Stage 3 (our meter-selection
adaptation) tests. Stage 1 (electrical distance formula) is tested
separately in tests/test_electrical_distance.py.
"""

from collections import Counter

import pytest

from core.ieee123_model import extract_all
from placement.feeder_context import FeederContext
from placement.feeder_adapter import network_and_loads_from_core_extraction
from placement.electrical_distance import get_system_y_index, build_electrical_distance_graph
from placement.community_detection import (
    louvain_communities,
    modularity,
    relabel_communities,
    merge_communities,
)
from placement.placement_strategy import CommunityDetectionPlacement


@pytest.fixture(scope="module")
def context_and_loads():
    extraction = extract_all()
    network, loads = network_and_loads_from_core_extraction(extraction)
    context = FeederContext(network, loads)
    context.annotate_loads(loads)
    return context, loads


@pytest.fixture(scope="module")
def electrical_adjacency(context_and_loads):
    context, _ = context_and_loads
    Y, node_index = get_system_y_index()
    adj = build_electrical_distance_graph(context, Y, node_index)
    return {i: {j: float(w) for j, w in nbrs.items()} for i, nbrs in adj.items()}


# --------------------------------------------------
# Ambiguity (c): weight = d_ij vs weight = 1/d_ij, tested not assumed
# --------------------------------------------------

def test_d_ij_weighting_beats_inverse_weighting(electrical_adjacency):
    """RES.md task spec ambiguity (c): the paper's prose and its own
    arithmetic disagree on whether larger d_ij means tighter or looser
    coupling. Empirical test: weight=d_ij should produce a materially
    better (higher-modularity) and more balanced partition than
    weight=1/d_ij, which collapses into one dominant community."""

    adj_d = electrical_adjacency
    adj_inv = {
        i: {j: (1.0 / w if w > 0 else 0.0) for j, w in nbrs.items()}
        for i, nbrs in adj_d.items()
    }

    comm_d = relabel_communities(louvain_communities(adj_d))
    comm_inv = relabel_communities(louvain_communities(adj_inv))

    q_d = modularity(adj_d, comm_d)
    q_inv = modularity(adj_inv, comm_inv)

    assert q_d > q_inv

    sizes_d = Counter(comm_d.values())
    sizes_inv = Counter(comm_inv.values())

    # weight=1/d_ij produces one community that swallows a large
    # fraction of the whole feeder (observed: 53 of 132 nodes, 40%);
    # weight=d_ij does not produce anything close to that.
    assert max(sizes_d.values()) / len(comm_d) < 0.30
    assert max(sizes_inv.values()) / len(comm_inv) >= 0.30


# --------------------------------------------------
# Stage 2: community detection stability
# --------------------------------------------------

def test_communities_are_deterministic(electrical_adjacency):
    a = relabel_communities(louvain_communities(electrical_adjacency))
    b = relabel_communities(louvain_communities(electrical_adjacency))

    # partition (grouping), not label identity, must match
    def as_partition(community_of):
        groups = {}
        for node, c in community_of.items():
            groups.setdefault(c, set()).add(node)
        return set(frozenset(g) for g in groups.values())

    assert as_partition(a) == as_partition(b)


def test_community_count_and_sizes_are_stable(electrical_adjacency):
    """Regression pin on the current result (standard Louvain,
    resolution=1.0, weight=d_ij) -- see
    reports/community_detection_report.txt for the full comparison
    against XuIoT26's ~4 subareas of ~30 nodes. Our partition is finer
    (documented: we omit eq. 7-9's scale-equalisation step)."""

    comm = relabel_communities(louvain_communities(electrical_adjacency))
    sizes = sorted(Counter(comm.values()).values(), reverse=True)

    assert len(sizes) == 12
    assert sizes == [24, 21, 19, 18, 18, 8, 8, 6, 5, 3, 1, 1]


def test_named_boundary_nodes_partially_reproduce_paper(electrical_adjacency):
    """XuIoT26 Section III-B names boundary nodes 18 (subareas I/II), 67
    (I/IV), 76 (III/IV). Our partition reproduces 18 and 67 as boundary
    nodes (different community than at least one topological neighbour)
    but not 76 -- reported, not forced."""

    comm = relabel_communities(louvain_communities(electrical_adjacency))

    def is_boundary(node):
        own = comm[node]
        return any(comm[n] != own for n in electrical_adjacency.get(node, {}))

    assert is_boundary("18") is True
    assert is_boundary("67") is True
    assert is_boundary("76") is False


# --------------------------------------------------
# Stage 3: our meter-selection adaptation
# --------------------------------------------------

def test_budget_allocation_sums_to_target():
    community_kw = {0: 1000.0, 1: 500.0, 2: 333.0, 3: 1.0}
    for target in (1, 5, 10, 49, 85):
        alloc = CommunityDetectionPlacement._allocate_budget_by_kw(community_kw, target)
        assert sum(alloc.values()) == target
        assert all(v >= 0 for v in alloc.values())


def test_budget_allocation_proportional_to_kw():
    community_kw = {"big": 900.0, "small": 100.0}
    alloc = CommunityDetectionPlacement._allocate_budget_by_kw(community_kw, 10)
    assert alloc["big"] == 9
    assert alloc["small"] == 1


@pytest.mark.parametrize("target", [15, 30, 44, 45, 49, 60, 70, 80, 85])
def test_community_placement_honours_count_and_ceiling(context_and_loads, target):
    context, loads = context_and_loads
    strategy = CommunityDetectionPlacement(target_meters=target)
    selected = strategy.select([dict(l) for l in loads], context)

    assert len(selected) == target
    assert len({l["bus"] for l in selected}) == target


def test_community_placement_above_ceiling_raises():
    with pytest.raises(ValueError):
        CommunityDetectionPlacement(target_meters=86)


def test_community_placement_is_deterministic(context_and_loads):
    context, loads = context_and_loads
    a = CommunityDetectionPlacement(target_meters=49).select([dict(l) for l in loads], context)
    b = CommunityDetectionPlacement(target_meters=49).select([dict(l) for l in loads], context)
    assert sorted(l["bus"] for l in a) == sorted(l["bus"] for l in b)


def test_community_placement_allocates_across_multiple_communities(context_and_loads):
    """Sanity check that this strategy is actually using community
    structure, not degenerating into a single-community selection."""

    context, loads = context_and_loads
    strategy = CommunityDetectionPlacement(target_meters=49)
    selected = strategy.select([dict(l) for l in loads], context)

    comm_of = strategy._community_of
    communities_used = {comm_of.get(str(l["bus"])) for l in selected}
    assert len(communities_used) >= 5


# --------------------------------------------------
# Merged variant (OURS -- not XuIoT26's; approximates the omitted
# eq. 7-9 scale-equalisation step by agglomerative merge)
# --------------------------------------------------

def test_merge_reduces_to_exact_target_count(electrical_adjacency):
    raw = relabel_communities(louvain_communities(electrical_adjacency))
    for target in (1, 2, 4, 8, 12):
        merged = merge_communities(electrical_adjacency, raw, target)
        assert len(set(merged.values())) == target


def test_merge_is_deterministic(electrical_adjacency):
    raw = relabel_communities(louvain_communities(electrical_adjacency))
    a = merge_communities(electrical_adjacency, raw, 4)
    b = merge_communities(electrical_adjacency, raw, 4)
    assert a == b


def test_merge_never_increases_community_count(electrical_adjacency):
    raw = relabel_communities(louvain_communities(electrical_adjacency))
    merged = merge_communities(electrical_adjacency, raw, 4)
    assert len(set(merged.values())) <= len(set(raw.values()))


def test_merged_variant_result_is_pinned():
    """Regression pin on the merged-to-4 result -- see
    reports/community_detection_report.txt. Notably unbalanced (93/37/1/1,
    not the paper's ~30-each): greedy total-weight merging concentrates
    growth in the already-largest community rather than balancing sizes.
    Node 18 remains a boundary node; 67 and 76 do NOT -- 67 actually
    LOSES boundary status it had in the raw 12-community variant. This
    is the reportable finding, not something to tune away."""

    CommunityDetectionPlacement.clear_cache()

    extraction = extract_all()
    network, loads = network_and_loads_from_core_extraction(extraction)
    context = FeederContext(network, loads)
    context.annotate_loads(loads)

    strategy = CommunityDetectionPlacement(target_meters=49, num_communities=4)
    comm = strategy._compute_communities(context)
    _, adjacency = CommunityDetectionPlacement._get_raw_communities(context)

    sizes = sorted(Counter(comm.values()).values(), reverse=True)
    assert sizes == [93, 37, 1, 1]

    def is_boundary(node):
        own = comm[node]
        return any(comm.get(n) != own for n in adjacency.get(node, {}))

    assert is_boundary("18") is True
    assert is_boundary("67") is False
    assert is_boundary("76") is False

    CommunityDetectionPlacement.clear_cache()


def test_both_variants_independently_selectable(context_and_loads):
    context, loads = context_and_loads
    CommunityDetectionPlacement.clear_cache()

    raw_strategy = CommunityDetectionPlacement(target_meters=49, num_communities=None)
    merged_strategy = CommunityDetectionPlacement(target_meters=49, num_communities=4)

    raw_comm = raw_strategy._compute_communities(context)
    merged_comm = merged_strategy._compute_communities(context)

    assert len(set(raw_comm.values())) == 12
    assert len(set(merged_comm.values())) == 4
    # both drew from the same cached raw Louvain result underneath
    assert raw_strategy._community_of != merged_strategy._community_of

    CommunityDetectionPlacement.clear_cache()
