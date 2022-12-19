from typing import Tuple, Iterable

from lib.a_star import AStar

NodeType = Tuple[int, int]


def dummy_heuristic(node) -> float:
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
    START_NODE: NodeType = (0, 0)
    END_NODE: NodeType = (10, 10)


    def manhattan_distance(node: NodeType) -> float:
        xa, ya = node
        xb, yb = END_NODE

        return abs(xa - xb) + abs(ya - yb)


    def is_goal(node: NodeType) -> bool:
        return node == END_NODE


    # choose your heuristic here:
    # HEURISTIC = dummy_heuristic
    HEURISTIC = manhattan_distance

    a_star: AStar[NodeType] = AStar(neighbor_function=neighbor_function)

    value, path = a_star.run(start=START_NODE, is_goal_function=is_goal, heuristic_function=HEURISTIC, debug=True)

    for d in a_star.debug_data:
        print(f"Opened {d.node} in distance {d.g}, with predecessor {d.predecessor}.")

    print()
    print(f"Optimal length: {value}")
    print(f"Optimal path: {path}")
