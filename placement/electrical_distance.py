"""
Equivalent electrical distance between buses, per XuIoT26 (Xu, Cao, Liu,
Wu, Hu, Zhang, Wu, Yu, "A Multiarea Data Reconstruction Framework to
Mitigate FDIA in IoT-Enabled Power Distribution Systems," IEEE IoT J.
13(14):31737-31751, 2026), Section II-B eq. (4)-(6):

    d^{mu,eta}_ij = |Y^{mu,eta}_ij| / |Y^{eta,eta}_jj|
                  + |Y^{mu,eta}_ij| / |Y^{mu,mu}_ii|                (4)-(5)

    d_ij = sum over mu,eta in {a,b,c} of d^{mu,eta}_ij                (6)

Y comes directly from OpenDSS's SystemY (dss.YMatrix.getYsparse() +
dss.Circuit.YNodeOrder()) on the compiled, solved circuit -- never from
a text parse. Requires the circuit to already be compiled and solved
(core.ieee123_model.compile_and_solve() / extract_all()).

VERIFIED against XuIoT26's worked examples (eq. 33-38), see
tests/test_electrical_distance.py:
  - three-phase line, buses 1-7: D_17 matrix and sum (4.101 vs paper's
    rounded 4.09; diagonal share 63.49% vs paper's 63.6%) match to
    rounding -- our unmodified IEEE-123 vs. the paper's DG-modified one.
  - single-phase (phase b) line, buses 1-2: d^{bb}_12 = 1.3078 vs
    paper's 1.31, all other phase-pair entries exactly 0.0, matches.
"""

import scipy.sparse as sp
import opendssdirect as dss


def get_system_y_index():
    """(Y: NxN complex csc_matrix, node_index: {'bus.phase' (lowercase): row/col})
    on the currently compiled+solved circuit."""

    data, indices, indptr = dss.YMatrix.getYsparse()
    n = len(indptr) - 1
    Y = sp.csc_matrix((data, indices, indptr), shape=(n, n))

    order = dss.Circuit.YNodeOrder()
    node_index = {name.lower(): i for i, name in enumerate(order)}

    return Y, node_index


def phase_pair_distance(Y, node_index, bus_i, phase_mu, bus_j, phase_eta):
    """d^{mu,eta}_ij. Returns 0.0 if either phase-node doesn't exist
    (bus doesn't carry that phase) or has zero self-admittance -- "for
    buses with no direct connection or coupling: d_ij = 0" (task spec,
    matching the paper's own stated convention)."""

    idx_i = node_index.get(f"{bus_i}.{phase_mu}")
    idx_j = node_index.get(f"{bus_j}.{phase_eta}")

    if idx_i is None or idx_j is None:
        return 0.0

    y_ij = abs(Y[idx_i, idx_j])
    y_ii = abs(Y[idx_i, idx_i])
    y_jj = abs(Y[idx_j, idx_j])

    if y_ii == 0.0 or y_jj == 0.0:
        return 0.0

    return y_ij / y_jj + y_ij / y_ii


def bus_pair_distance_matrix(Y, node_index, bus_i, bus_j, phases=(1, 2, 3)):
    """3x3 matrix D[mu-1][eta-1] = d^{mu,eta}_ij, row=phase of bus_i,
    col=phase of bus_j (this indexing was verified against XuIoT26's
    worked D_17 example -- see module docstring)."""

    return [
        [phase_pair_distance(Y, node_index, bus_i, mu, bus_j, eta) for eta in phases]
        for mu in phases
    ]


def bus_pair_distance(Y, node_index, bus_i, bus_j, phases=(1, 2, 3)):
    """d_ij, eq. (6): sum of all 9 phase-pair entries."""

    matrix = bus_pair_distance_matrix(Y, node_index, bus_i, bus_j, phases)
    return sum(sum(row) for row in matrix)


def build_electrical_distance_graph(context, Y, node_index):
    """Weighted adjacency {bus: {neighbor_bus: d_ij}}, computed over
    context.graph's edges (the RAW, un-collapsed bus graph -- every
    edge there is a real branch element, i.e. guaranteed direct Y-bus
    coupling; d_ij is 0 by construction for any non-adjacent pair, so
    there is nothing to gain from computing the full N^2 bus-pair
    matrix). This deliberately includes regulator-secondary buses
    (150r, 9r, 25r, 160r) and the substation/open-tie helper buses as
    graph nodes -- they are real Y-bus nodes and, given the near-zero
    regulator impedance in this model, end up extremely tightly coupled
    to their primary-side bus (very large d_ij), so Louvain naturally
    clusters them with it without any special-casing. They host no
    Load objects, so Stage 3 (meter selection) never selects them."""

    adjacency = {}
    seen_pairs = set()

    for bus_i, neighbours in context.graph.items():
        for bus_j in neighbours:
            pair = frozenset((bus_i, bus_j))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            d_ij = bus_pair_distance(Y, node_index, bus_i, bus_j)

            adjacency.setdefault(bus_i, {})[bus_j] = d_ij
            adjacency.setdefault(bus_j, {})[bus_i] = d_ij

    return adjacency
