from collections import defaultdict

# https://benalexkeen.com/implementing-djikstras-shortest-path-algorithm-with-python/

routes = "Tatooine-Dagobah:6, Dagobah-Endor:4, Dagobah-Hoth:1, Hoth-Endor:1, Tatooine-Hoth:6"
autonomy = 6
countdown = 7


class Graph:
    def __init__(self):
        self.edges = defaultdict(list)
        self.weights = {}

    def add_edge(self, from_node, to_node, weight):
        self.edges[from_node].append(to_node)
        self.edges[to_node].append(from_node)
        self.weights[(from_node, to_node)] = weight
        self.weights[(to_node, from_node)] = weight


graph = Graph()
edges = []

for route in routes.split(","):
    distance = int(route[-1])
    src_dst = route.split(":")[0]
    src = src_dst.split("-")[0].strip()
    dst = src_dst.split("-")[1].strip()
    edges.append((src, dst, distance))

for edge in edges:
    graph.add_edge(*edge)


def dijsktra(graph, initial, end):
    # shortest paths is a dict of nodes
    # whose value is a tuple of (previous node, weight)
    shortest_paths = {initial: (None, 0)}
    current_node = initial
    visited = set()

    while current_node != end:
        visited.add(current_node)
        destinations = graph.edges[current_node]
        weight_to_current_node = shortest_paths[current_node][1]

        for next_node in destinations:
            weight = graph.weights[(current_node, next_node)] + weight_to_current_node
            if next_node not in shortest_paths:
                shortest_paths[next_node] = (current_node, weight)
            else:
                current_shortest_weight = shortest_paths[next_node][1]
                if current_shortest_weight > weight:
                    shortest_paths[next_node] = (current_node, weight)

        next_destinations = {node: shortest_paths[node] for node in shortest_paths if node not in visited}
        if not next_destinations:
            return "Route Not Possible"
        # next node is the destination with the lowest weight
        current_node = min(next_destinations, key=lambda k: next_destinations[k][1])

    # print(shortest_paths)
    # Work back through destinations in shortest path
    path = []
    path_distance = shortest_paths[current_node][1]
    while current_node is not None:
        path.append(current_node)
        next_node = shortest_paths[current_node][0]
        current_node = next_node
    # Reverse path
    path = path[::-1]
    # print(shortest_paths)
    return path, path_distance


shortest_path, minimum_distance = dijsktra(graph, 'Tatooine', 'Endor')
print(shortest_path, minimum_distance)
if countdown > minimum_distance:
    print("100")
elif minimum_distance > autonomy:
    print("0")
