from abc import ABC, abstractmethod
from collections import Counter, defaultdict, deque
import math
from collections import defaultdict
from pickle import load

class PlacementStrategy(ABC):

    @abstractmethod
    def select(self, loads, network):
        pass


class HybridRepresentativePlacement(PlacementStrategy):

    def __init__(self, target_meters=49):
        self.target_meters = target_meters

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

   
    def select(self, loads, network):

        loads = self._assign_size_classes(loads)

        

        utility_nodes = self._get_utility_nodes(network)
        #debug to see if utility nodes are being identified correctly
        # print("\n===== UTILITY NODES =====")
        # for node in sorted(utility_nodes):
        #     print(node)


        graph = self._build_graph(network)

        utility_nodes = self._get_utility_nodes(network)

        topology_graph = self._build_topology_graph(
            graph,
            utility_nodes
        )

        parent, children = self._root_tree(
            topology_graph,
            str(network["source"]["bus"])
        )

        #debug
        print("\n===== TOPOLOGY GRAPH =====")

        for node in sorted(utility_nodes):
            print(f"\n{node}")

            print("Neighbours:", graph[node])

            if node in topology_graph:
                print("Topology:", topology_graph[node])
            else:
                print("Removed from topology (helper node)")

            for nbr in graph[node]:
                print(f"    {nbr} -> {graph[nbr]}")



        bus_load = self._build_bus_loads(loads)
        #debug
        print("\n===== BUS LOADS =====")
        for bus in sorted(bus_load.keys(), key=int):
            print(f"Bus {bus:>3}: {bus_load[bus]:>6.1f} kW")

        downstream_load = {}
        self._compute_downstream_load(str(network["source"]["bus"]),children,bus_load,downstream_load)
        #debug
        print("\n===== TOP DOWNSTREAM LOADS =====")
        for node, load in sorted(
            downstream_load.items(),
            key=lambda x: x[1],
            reverse=True
        )[:20]:
            print(f"{node:>6} : {load:8.1f} kW")

        main_feeder = self._extract_main_feeder(str(network["source"]["bus"]),children,downstream_load)
        #debug
        print("\n===== MAIN FEEDER =====")
        for i, bus in enumerate(main_feeder):
            print(f"{i:2d}: {bus}")
        print("\n===== CHECK FOR HELPER NODES =====")
        for bus in main_feeder:
            if (
                bus.endswith("r") or
                bus.endswith("s") or
                "_OPEN" in bus
            ):
                print("Helper node found:", bus)

        main_feeder_set = set(main_feeder)

        max_downstream = max(downstream_load.values())

        for load in loads:

            bus = str(load["bus"])

            load["downstream_kw"] = downstream_load.get(bus, 0)

            load["downstream_score"] = (
                load["downstream_kw"] / max_downstream
            )

            load["on_backbone"] = (
                bus in main_feeder_set
            )

        self._compute_depths(loads, topology_graph, str(network["source"]["bus"]),network)
        # #debug to chk if depth is being used
        # for load in loads[:10]:
        #     print(
        #         load["name"],
        #         load["bus"],
        #         load["depth"]
        #     )
 

        self._assign_branch_ids(loads, topology_graph, str(network["source"]["bus"]), network)
        #debug to ck if brnch is used
        for load in loads:
            print(
                load["name"],
                load["bus"],
                load["branch"]
            )
 

        print("\n===== SOURCE =====")
        print(network["source"])
        print("\n===== REGULATORS =====")
        for reg in network["assets"]["regulators"]:
            print(reg)

        self._assign_scores(loads)

        selected = self._representative_selection(loads)

        selected = self._balance_phases(selected, loads)

        selected = self._balance_connections(selected, loads)

        return selected

    # --------------------------------------------------
    # Stage 1
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

    # --------------------------------------------------
    # Stage 2
    # --------------------------------------------------

    def _build_graph(self, network):

        graph = defaultdict(set)

        for edge in network["edges"]:

            a = edge["from"]
            b = edge["to"]

            graph[a].add(b)
            graph[b].add(a)


        return graph


 

    def _build_topology_graph(self, graph, utility_nodes):

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

            # Ignore substation helper (61s) and OPEN switch helpers
            # They simply disappear.

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

    def _root_tree(self, graph, source):
        """
        Convert the undirected feeder graph into
        a rooted tree using BFS.
        """

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

    def _build_bus_loads(self, loads):
        """
        Aggregate all loads connected to the same bus.

        Returns
        -------
        dict
            {
                "1": 40.0,
                "2": 20.0,
                "49": 120.0,
                ...
            }
        """

        bus_load = {}

        for load in loads:

            bus = str(load["bus"])

            bus_load.setdefault(bus, 0.0)

            bus_load[bus] += load["base_kw"]

        return bus_load

    def _compute_downstream_load(self, node, children, bus_load, downstream):
        """
        Recursively compute cumulative downstream load.

        Parameters
        ----------
        node : str
            Current node ID.

        children : dict
            Rooted tree.

        bus_load : dict
            Local load at each bus.x`

        downstream : dict
            Output dictionary.
        """

        total = bus_load.get(node, 0.0)

        for child in children.get(node, []):

            total += self._compute_downstream_load(
                child,
                children,
                bus_load,
                downstream
            )

        downstream[node] = total

        return total


    def _extract_main_feeder(self,source,children,downstream_load):
        """
        Follow the child with the largest downstream
        cumulative load until a leaf is reached.
        """

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

    def _get_utility_nodes(self, network):
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
    

    def _compute_depths(self, loads, graph, source,network):
        source = str(source)#fix
        depth = {source: 0}

        q = deque([source])
        # #debug stmts
        # print("Source:", source)
        # print(type(source))
        # print(type(next(iter(graph.keys()))))

        # print("150 :", graph.get("150"))
        # print("150r:", graph.get("150r"))
        # print("149 :", graph.get("149"))

        # print("Neighbors of source:", graph.get(source))

        while q:

            node = q.popleft()

            for nbr in graph[node]:

                if nbr not in depth:
                    depth[nbr] = depth[node] + 1
                    q.append(nbr)

        # Map base bus number -> actual node id in the graph else 999 if node not in graph
            for load in loads:

                bus = str(load["bus"])

                load["depth"] = depth.get(bus, 999)



    def _assign_branch_ids(self, loads, graph, source, network):

        source = str(source)

        branch = {}

        visited = {source}

        q = deque([(source, 0)])      # (node, branch_id)

        next_branch = 1

        while q:

            node, current_branch = q.popleft()

            branch[node] = current_branch

            children = [
                nbr
                for nbr in graph[node]
                if nbr not in visited
            ]

            # Continue on same branch
            if len(children) <= 1:

                for child in children:

                    visited.add(child)

                    q.append((child, current_branch))

            # Split into new branches
            else:

                for child in children:

                    visited.add(child)

                    q.append((child, next_branch))

                    next_branch += 1

        # Assign every load a branch id
        for load in loads:

            bus = str(load["bus"])

            load["branch"] = branch.get(bus, -1)

    # --------------------------------------------------
    # Stage 3
    # --------------------------------------------------

    
    def _assign_scores(self, loads):

        max_depth = max(load["depth"] for load in loads)

        for load in loads:

            # -------------------------
            # Load size
            # -------------------------

            kw_score = min(load["base_kw"] / 100.0, 1.0)

            # -------------------------
            # Electrical depth
            # -------------------------

            depth_score = load["depth"] / max_depth

            # -------------------------
            # Downstream importance
            # -------------------------

            downstream_score = load["downstream_score"]

            # -------------------------
            # Backbone bonus
            # -------------------------

            backbone_bonus = (
                1.5 if load["on_backbone"] else 0.0
            )

            load["score"] = (

                0.30 * kw_score +

                0.20 * depth_score +

                0.40 * downstream_score +

                0.20 * backbone_bonus

            )

    # --------------------------------------------------
    # Stage 4
    # --------------------------------------------------

    def _representative_selection(self, loads):

        groups = defaultdict(list)

        for load in loads:
            groups[load["size_class"]].append(load)

        total = len(loads)

        selected = []

        for size_class, items in groups.items():

            quota = max(
                1,
                round(self.target_meters * len(items) / total)
            )

            # ----------------------------------
            # Group by branch
            # ----------------------------------

            branches = defaultdict(list)

            for load in items:
                branches[load["branch"]].append(load)

            # Sort each branch individually
            for branch in branches.values():
                branch.sort(
                    key=lambda x: x["score"],
                    reverse=True
                )

            used_buses = {load["bus"] for load in selected}

            count = 0

            # ----------------------------------
            # Round-robin across branches
            # ----------------------------------

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

            selected.sort(
                key=lambda x: x["score"],
                reverse=True
            )

            selected = selected[:self.target_meters]

        return selected
    # --------------------------------------------------
    # Stage 5
    # --------------------------------------------------

    def _balance_phases(self, selected, all_loads):

        # Placeholder
        return selected

    def _balance_connections(self, selected, all_loads):

        # Placeholder
        return selected

    # --------------------------------------------------

    def _compute_quartiles(self, values):

        n = len(values)

        def percentile(p):

            index = (n - 1) * p

            lower = math.floor(index)
            upper = math.ceil(index)

            if lower == upper:
                return values[lower]

            fraction = index - lower

            return (
                values[lower]
                + (values[upper] - values[lower]) * fraction
            )

        return (
            percentile(0.25),
            percentile(0.50),
            percentile(0.75),
        )