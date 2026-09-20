"""
Community detection for meter placement, adapted from XuIoT26 Section
II-B, eq. (7)-(10), with two documented simplifications (see module
docstring below and CommunityDetectionPlacement in placement_strategy.py).

XuIoT26's own text has three problems that block a literal
implementation:

  (a) eq. (9) uses a term omega_ij that is never defined anywhere in
      the paper.
  (b) eq. (10) is written over A_ij and kappa_i*kappa_j (node-graph
      quantities) while the surrounding prose says F (the edge graph
      from eq. 7-9) is the adjacency matrix being maximised over --
      these are inconsistent as written.
  (c) The prose says "the tighter the connection, the smaller the
      equivalent electrical distance and the larger the edge weight,"
      but the arithmetic in the paper's OWN worked example says the
      opposite: d_17 = 4.09 for a three-phase (tightly coupled) line
      vs. d_12 = 1.31 for a single-phase (loosely coupled) line -- a
      LARGER d_ij for the TIGHTER connection. d_ij behaves as a
      similarity, not a distance, and A_ij = d_ij is used directly as
      the modularity edge weight. Verified empirically below (see
      test_community_detection.py: weight=d_ij produces sensible
      communities, weight=1/d_ij does not).

Given (a) and (b), this implements standard weighted Louvain modularity
maximisation directly on the NODE graph (bus graph, A_ij = d_ij), not
on the edge-graph transformation of eq. (7)-(9). XuIoT26 states the
edge-graph transformation's purpose is "scale equalization of each
subregion" -- a refinement to balance subarea sizes, not the core
partitioning mechanism -- so omitting it is a real simplification, not
a silent one, and is documented here and in
reports/community_detection_report.txt.

Two variants are exposed (both are real, both are kept -- see
CommunityDetectionPlacement):
  - the RAW Louvain result (12 communities on this feeder) -- the
    faithful output of the equations we CAN implement.
  - a MERGED variant (merge_communities(), below) -- OUR adaptation,
    agglomeratively merging the raw communities down to a target count
    (default 4, matching XuIoT26's reported subarea count) by strongest
    inter-community coupling. This approximates the effect of the
    omitted eq.(7)-(9) step; it is not a re-derivation of it.
"""

from collections import defaultdict


def _weighted_degree(adjacency, node):
    return sum(adjacency.get(node, {}).values())


def total_weight(adjacency):
    """m = sum of all edge weights, each undirected edge counted once."""

    total = 0.0
    seen = set()
    for i, neighbours in adjacency.items():
        for j, w in neighbours.items():
            pair = frozenset((i, j))
            if pair in seen:
                continue
            seen.add(pair)
            total += w
    return total


def modularity(adjacency, communities, resolution=1.0):
    """Reichardt-Bornholdt generalised weighted modularity:
    Q = (1/2m) * sum_ij [A_ij - resolution*k_i*k_j/2m] * delta(c_i, c_j).
    resolution=1.0 is standard Newman modularity. resolution<1 relaxes
    the well-known modularity resolution limit, favouring fewer/larger
    communities -- a standard, textbook Louvain parameter, not a new
    mechanism. See CommunityDetectionPlacement's docstring for why this
    is used here (recovering a size scale closer to XuIoT26's ~4
    subareas without the omitted eq. 7-9 scale-equalisation step)."""

    m2 = 2.0 * total_weight(adjacency)
    if m2 == 0:
        return 0.0

    degree = {node: _weighted_degree(adjacency, node) for node in adjacency}

    q = 0.0
    for i, neighbours in adjacency.items():
        for j, w in neighbours.items():
            if communities[i] == communities[j]:
                q += w - resolution * (degree[i] * degree[j]) / m2

    return q / m2


def _local_moving_pass(adjacency, community_of, resolution=1.0):
    """One Louvain local-moving phase: repeatedly try moving each node
    into the neighbouring community that gives the largest positive
    modularity gain, until no move improves modularity. Returns True if
    any node moved."""

    m = total_weight(adjacency)
    if m == 0:
        return False

    degree = {node: _weighted_degree(adjacency, node) for node in adjacency}

    # Sigma_tot[c] = sum of weighted degrees of nodes currently in community c
    sigma_tot = defaultdict(float)
    for node, c in community_of.items():
        sigma_tot[c] += degree[node]

    any_move = True
    moved_at_all = False

    while any_move:
        any_move = False

        for node in adjacency:
            current_c = community_of[node]
            k_i = degree[node]

            # weight from node to each neighbouring community
            neighbour_weight = defaultdict(float)
            for nbr, w in adjacency[node].items():
                if nbr == node:
                    continue
                neighbour_weight[community_of[nbr]] += w

            # remove node from its current community
            sigma_tot[current_c] -= k_i

            best_c = current_c
            best_gain = neighbour_weight.get(current_c, 0.0) / m - resolution * (
                sigma_tot[current_c] * k_i
            ) / (2.0 * m * m)

            for c, k_i_in in neighbour_weight.items():
                gain = k_i_in / m - resolution * (sigma_tot[c] * k_i) / (2.0 * m * m)
                if gain > best_gain:
                    best_gain = gain
                    best_c = c

            sigma_tot[best_c] += k_i
            if best_c != current_c:
                community_of[node] = best_c
                any_move = True
                moved_at_all = True

    return moved_at_all


def _aggregate(adjacency, community_of):
    """Build the community-level graph: one node per community, edge
    weight = sum of inter-community edge weights, self-loop weight =
    sum of intra-community edge weights (standard Louvain aggregation,
    self-loops counted once per undirected edge)."""

    aggregated = defaultdict(lambda: defaultdict(float))
    seen = set()

    for i, neighbours in adjacency.items():
        for j, w in neighbours.items():
            pair = frozenset((i, j))
            if pair in seen:
                continue
            seen.add(pair)

            ci, cj = community_of[i], community_of[j]
            aggregated[ci][cj] += w
            if ci != cj:
                aggregated[cj][ci] += w

    return {k: dict(v) for k, v in aggregated.items()}


def louvain_communities(adjacency, max_levels=10, resolution=1.0):
    """Multi-level Louvain modularity maximisation. Returns
    {original_node: community_id} (community IDs are arbitrary but
    stable within one call)."""

    if not adjacency:
        return {}

    # Level 0: original nodes, each its own community
    community_of = {node: node for node in adjacency}
    current_graph = adjacency
    # node_to_level0: for the current (possibly aggregated) graph,
    # maps its node ids back to the ORIGINAL node ids they represent
    node_to_level0 = {node: {node} for node in adjacency}

    for _level in range(max_levels):
        level_community = {node: node for node in current_graph}
        moved = _local_moving_pass(current_graph, level_community, resolution)

        if not moved:
            break

        # Aggregate: build next-level graph from this level's communities
        next_graph = _aggregate(current_graph, level_community)

        # Update node_to_level0: community c at this level absorbs all
        # original nodes previously mapped to its members
        next_node_to_level0 = defaultdict(set)
        for node, c in level_community.items():
            next_node_to_level0[c] |= node_to_level0[node]

        current_graph = next_graph
        node_to_level0 = dict(next_node_to_level0)

        if len(current_graph) <= 1:
            break

    # Flatten: assign each original node the id of the top-level
    # community that contains it
    result = {}
    for community_id, members in node_to_level0.items():
        for node in members:
            result[node] = community_id

    # Any node never touched by aggregation (single-level graph, no
    # moves at all) keeps its own id as its community
    for node in adjacency:
        result.setdefault(node, node)

    return result


def merge_communities(adjacency, community_of, target_count):
    """OURS -- no XuIoT26 source. Agglomeratively merge communities by
    total inter-community edge weight (sum of d_ij across the boundary
    between each pair), merging the strongest-coupled pair each step,
    until only target_count communities remain.

    This exists because eq. (7)-(9)'s edge-graph transformation --
    whose stated purpose is "scale equalization of each subregion" --
    could not be implemented (eq. 9's omega_ij is undefined in the
    source). Merging the as-computed Louvain communities down to
    XuIoT26's reported subarea count (4, on their DG-modified IEEE-123)
    is our own adaptation to approximate that missing step's effect,
    not a re-derivation of it. See
    reports/community_detection_report.txt."""

    community_of = dict(community_of)

    def inter_community_weights():
        weights = defaultdict(float)
        seen = set()
        for i, neighbours in adjacency.items():
            for j, w in neighbours.items():
                pair_nodes = frozenset((i, j))
                if pair_nodes in seen:
                    continue
                seen.add(pair_nodes)

                ci, cj = community_of[i], community_of[j]
                if ci == cj:
                    continue
                weights[frozenset((ci, cj))] += w
        return weights

    current_count = len(set(community_of.values()))

    while current_count > target_count:
        weights = inter_community_weights()
        if not weights:
            break  # remaining communities share no edge at all -- stop

        strongest_pair = max(weights, key=weights.get)
        c1, c2 = tuple(strongest_pair)

        for node, c in community_of.items():
            if c == c2:
                community_of[node] = c1

        current_count -= 1

    return community_of


def relabel_communities(community_of):
    """Community IDs from louvain_communities() are original node ids
    (arbitrary strings) -- relabel to 0..k-1 for readability."""

    unique = sorted(set(community_of.values()), key=str)
    remap = {old: i for i, old in enumerate(unique)}
    return {node: remap[c] for node, c in community_of.items()}
