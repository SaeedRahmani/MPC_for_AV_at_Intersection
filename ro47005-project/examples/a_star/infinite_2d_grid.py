import sys
from typing import Tuple, Iterable

from lib.a_star import AStar

NodeType = Tuple[int, int]


def manhattan_distance(node: NodeType, end: NodeType) -> float:
    xa, ya = node
    xb, yb = end

    return abs(xa - xb) + abs(ya - yb)


def dummy_heuristic(node, end) -> float:
    # to turn A* into Dijkstra
    return 0


def neighbor_function(node: NodeType) -> Iterable[Tuple[float, NodeType]]:
    # for each node, returns the four nodes around it

    x, y = node
    for a, b in [[-1, 0], [0, -1], [1, 0], [0, 1]]:
        neighbor = x + a, y + b
        edge_cost = 1.

        yield edge_cost, neighbor


if __name__ == '__main__':
    # choose your heuristic here:
    # HEURISTIC = dummy_heuristic
    HEURISTIC = manhattan_distance

    START_NODE: NodeType = (0, 0)
    END_NODE: NodeType = (10, 10)

    a_star: AStar[NodeType] = AStar(neighbor_function=neighbor_function)

    value, path = a_star.run(START_NODE, END_NODE, heuristic_function=HEURISTIC, debug=True, debug_file=sys.stdout)

    print()
    print(f"Optimal length: {value}")
    print(f"Optimal path: {path}")
