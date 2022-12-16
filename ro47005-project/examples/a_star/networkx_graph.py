import sys
from typing import Callable, Tuple, Iterable

from lib.a_star import AStar
import networkx as nx

if __name__ == '__main__':
    # directed graph from Assignment 1 task 5
    g = nx.DiGraph()
    g.add_nodes_from(['Start', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'Goal'])
    g.add_edges_from([
        ('Start', 'A', {'weight': 3}),
        ('Start', 'B', {'weight': 1}),
        ('Start', 'D', {'weight': 4}),
        ('A', 'C', {'weight': 2}),
        ('B', 'D', {'weight': 5}),
        ('B', 'E', {'weight': 6}),
        ('B', 'H', {'weight': 1}),
        ('C', 'D', {'weight': 2}),
        ('C', 'F', {'weight': 1}),
        ('C', 'Goal', {'weight': 9}),
        ('D', 'L', {'weight': 2}),
        ('E', 'J', {'weight': 4}),
        ('F', 'D', {'weight': 1}),
        ('G', 'K', {'weight': 3}),
        ('H', 'G', {'weight': 4}),
        ('H', 'I', {'weight': 6}),
        ('H', 'O', {'weight': 2}),
        ('I', 'J', {'weight': 5}),
        ('J', 'G', {'weight': 3}),
        ('J', 'Goal', {'weight': 3}),
        ('K', 'N', {'weight': 1}),
        ('M', 'K', {'weight': 1}),
        ('M', 'Goal', {'weight': 2}),
        ('N', 'M', {'weight': 2}),
        ('O', 'L', {'weight': 2}),
    ])


    def create_neighbor_function_for_graph(g: nx.DiGraph) -> Callable[[str], Iterable[Tuple[float, str]]]:
        def neighbor_function(node: str) -> Iterable[Tuple[float, str]]:
            return ((data['weight'], n) for n, data in g[node].items())

        return neighbor_function


    def is_goal(node: str) -> bool:
        return node == 'Goal'


    neighbor_function = create_neighbor_function_for_graph(g)

    a_star: AStar[str] = AStar(neighbor_function=neighbor_function)

    value, path = a_star.run('Start', is_goal_function=is_goal, heuristic_function=lambda node: 0, debug=True, debug_file=sys.stdout)

    print()
    print(f"Optimal length: {value}")
    print(f"Optimal path: {path}")
