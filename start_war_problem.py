import heapq
import sys
import math

# Auto-generated code below aims at helping you parse
# the standard input according to the problem statement.
from collections import defaultdict
from typing import List

autonomy = input()
routes = input()
countdown = input()
bountyhunters = input()


class ShortestPath:
    def countPaths(self, n: int, roads: List[List[int]]) -> int:
        # build undirected graph by the roads, mapping the one city to the other city with times
        # use a heap to store (time, city)
        # use a dictionary to store the visited city
        # each time, pop out the top of the heap, expore the time to get to the next city
        # keep track of the ways to arrive each city
        # the number of ways to arrive current city depends on the number of ways to arrive previous city that leads to current city
        # if we neet the city before, but has greater time, skip it
        # if we meet the city before, but has the same time, add the number of previous ways to the current ways
        # if we meet the city before, but has the less time, update the number of ways as previous ways
        # O(nlogn) time and O(n) space

        graph = defaultdict(dict)

        for a, b, time in roads:
            graph[a][b] = time
            graph[b][a] = time

        pq = [(0, 0)]
        seen = [float("inf")] * n
        seen[0] = 0
        ways = [0] * n
        ways[0] = 1

        shortest = None
        while pq:
            time, city = heapq.heappop(pq)
            for nxt in graph[city]:
                newTime = time + graph[city][nxt]
                if seen[nxt] >= newTime:
                    if seen[nxt] > newTime:
                        seen[nxt] = newTime
                        ways[nxt] = ways[city]
                        heapq.heappush(pq, (newTime, nxt))
                    elif seen[nxt] == newTime:
                        ways[nxt] += ways[city]

        return ways[-1] % (10 ** 9 + 7)


ShortestPath.countPaths()

"Tatooine-Dagobah:6, Dagobah-Endor:4, Dagobah-Hoth:1, Hoth-Endor:1, Tatooine-Hoth:6"
routes_distance = {
    "route1": {"src": "Tatooine", "dst": "Dagobah", "distance": 6},
    "route2": {"src": "Dagobah", "dst": "Endor", "distance": 4},
    "route3": {"src": "Dagobah", "dst": "Hoth", "distance": 1},
    "route4": {"src": "Hoth", "dst": "Endor", "distance": 1},
    "route4": {"src": "Tatooine", "dst": "Hoth", "distance": 6}
}
# tatoonie -> dagobah -> endor => 10
# tatoonie -> dagobah -> Hoth -> Endor => 8
# tatooine -> hoth -> endor => 7

# Write an answer using print
# To debug: print("Debug messages...", file=sys.stderr, flush=True)

print("answer")
