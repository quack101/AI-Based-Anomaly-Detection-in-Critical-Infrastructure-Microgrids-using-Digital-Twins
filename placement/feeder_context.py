"""
Shared feeder-topology computation, extracted from the original
HybridRepresentativePlacement so every placement strategy (hybrid,
random, degree, community-detection) works from the same graph, rooted
tree, bus loads, downstream loads, main feeder, depths and branch IDs --
computed once, not per strategy.

The functions here are moved from parser/placement_strategy.py, not
rewritten: same graph-build, same regulator-helper-node collapse, same
rooted-tree BFS, same downstream-load recursion, same main-feeder
extraction, same branch-ID assignment. The only behavioural change is
_compute_depths / _assign_branch_ids no longer mutate a `loads` list
internally (that was Step-3 defect #5: the loads-annotation loop was
mis-indented inside the BFS while-loop, making it re-run on every node
visit). They now return plain depth/branch dicts; FeederContext.annotate_loads()
applies them once, to any loads list, on request.
"""

from collections import defaultdict, deque


def get_utility_nodes(network):
    """Regulator-secondary ('...r'), substation-transformer-secondary
    ('...s') and open-tie-switch stub ('..._OPEN') buses -- helper nodes
    collapsed out of the topology graph, not real feeder buses."""

    utility = set()

    for node in network["nodes"]:
        node_id = str(node["id"])

        if (
            node_id.endswith("r") or
            node_id.endswith("s") or
            "_OPEN" in node_id
        ):
            utility.add(node_id)

    return utility


def build_graph(network):
    graph = defaultdict(set)

    for edge in network["edges"]:
        a = edge["from"]
        b = edge["to"]

        graph[a].add(b)
        graph[b].add(a)

    return graph


def build_topology_graph(graph, utility_nodes):
    topology = defaultdict(set)

    # Create entries only for real buses
    for node in graph:
        if node in utility_nodes:
            continue
        topology[node]

    # Collapse regulator helper buses
    for helper in utility_nodes:
        if helper not in graph:
            continue

        neighbours = list(graph[helper])

        # Collapse only regulator helper nodes
        if helper.endswith("r") and len(neighbours) == 2:
            a, b = neighbours
            topology[a].add(b)
            topology[b].add(a)

        # Ignore substation helper (61s) and OPEN switch helpers --
        # they simply disappear.

    # Preserve direct real-bus connections
    for node in graph:
        if node in utility_nodes:
            continue

        for nbr in graph[node]:
            if nbr in utility_nodes:
                continue
            topology[node].add(nbr)

    return {
        node: sorted(neighbours)
        for node, neighbours in topology.items()
    }


def root_tree(graph, source):
    """Convert the undirected feeder graph into a rooted tree using BFS."""

    parent = {source: None}
    children = defaultdict(list)

    q = deque([source])

    while q:
        node = q.popleft()

        for nbr in graph[node]:
            if nbr in parent:
                continue

            parent[nbr] = node
            children[node].append(nbr)

            q.append(nbr)

    return parent, children


def build_bus_loads(loads):
    """Aggregate all loads connected to the same bus -> {"1": 40.0, ...}."""

    bus_load = {}

    for load in loads:
        bus = str(load["bus"])
        bus_load.setdefault(bus, 0.0)
        bus_load[bus] += load["base_kw"]

    return bus_load


def compute_downstream_load(node, children, bus_load, downstream):
    """Recursively compute cumulative downstream load, writing into
    `downstream` in place (matches the original recursion exactly)."""

    total = bus_load.get(node, 0.0)

    for child in children.get(node, []):
        total += compute_downstream_load(child, children, bus_load, downstream)

    downstream[node] = total

    return total


def extract_main_feeder(source, children, downstream_load):
    """Follow the child with the largest downstream cumulative load
    until a leaf is reached."""

    backbone = []
    current = source

    while True:
        backbone.append(current)

        if current not in children:
            break

        if len(children[current]) == 0:
            break

        current = max(
            children[current],
            key=lambda child: downstream_load.get(child, 0.0)
        )

    return backbone


def compute_depths(graph, source):
    """BFS depth of every node from `source`. Pure function -- returns a
    dict, does not touch any `loads` list (Step-3 defect #5 fix: this
    used to be interleaved with a loads-annotation loop mis-indented
    inside the BFS while-loop, making the annotation re-run on every
    node visit, O(V*L) instead of O(V+L))."""

    source = str(source)
    depth = {source: 0}

    q = deque([source])

    while q:
        node = q.popleft()

        for nbr in graph[node]:
            if nbr not in depth:
                depth[nbr] = depth[node] + 1
                q.append(nbr)

    return depth


def assign_branch_ids(graph, source):
    """BFS branch-ID assignment: a node keeps its parent's branch id
    unless the parent has more than one remaining child, in which case
    each child starts a new branch. Pure function -- returns a dict."""

    source = str(source)
    branch = {}
    visited = {source}

    q = deque([(source, 0)])
    next_branch = 1

    while q:
        node, current_branch = q.popleft()
        branch[node] = current_branch

        children = [nbr for nbr in graph[node] if nbr not in visited]

        if len(children) <= 1:
            for child in children:
                visited.add(child)
                q.append((child, current_branch))
        else:
            for child in children:
                visited.add(child)
                q.append((child, next_branch))
                next_branch += 1

    return branch


class FeederContext:
    """Feeder-topology computation, built once from (network, loads) and
    shared by every PlacementStrategy. Strategies that don't need
    downstream-load/branch/backbone semantics (random, degree,
    community-detection) may ignore those fields; they're still
    computed once, centrally, rather than recomputed per strategy."""

    def __init__(self, network, loads):
        self.network = network
        self.source = str(network["source"]["bus"])

        self.utility_nodes = get_utility_nodes(network)
        self.graph = build_graph(network)
        self.topology_graph = build_topology_graph(self.graph, self.utility_nodes)

        self.parent, self.children = root_tree(self.topology_graph, self.source)

        self.bus_load = build_bus_loads(loads)

        self.downstream_load = {}
        compute_downstream_load(
            self.source, self.children, self.bus_load, self.downstream_load
        )
        self.max_downstream = max(self.downstream_load.values())

        self.main_feeder = extract_main_feeder(
            self.source, self.children, self.downstream_load
        )
        self.main_feeder_set = set(self.main_feeder)

        self.depths = compute_depths(self.topology_graph, self.source)
        self.branch_ids = assign_branch_ids(self.topology_graph, self.source)

    def annotate_loads(self, loads):
        """Apply topology-derived fields to every load in `loads`, in
        place, and return it. Strategy-independent -- called once on
        the full load list before any strategy.select() runs."""

        for load in loads:
            bus = str(load["bus"])

            load["downstream_kw"] = self.downstream_load.get(bus, 0.0)
            load["downstream_score"] = (
                load["downstream_kw"] / self.max_downstream
                if self.max_downstream else 0.0
            )
            load["on_backbone"] = bus in self.main_feeder_set
            load["depth"] = self.depths.get(bus, 999)
            load["branch"] = self.branch_ids.get(bus, -1)

        return loads

    def bus_degree(self, bus):
        """Topology-graph degree of a bus -- number of adjacent real
        buses after helper-node collapse. Used by DegreePlacement."""

        return len(self.topology_graph.get(str(bus), []))
