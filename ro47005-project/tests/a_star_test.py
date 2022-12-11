import math
import unittest
from typing import Tuple, Iterable

from lib.a_star import AStar


def dummy_heuristic(node, end):
    # turns A* into Dijkstra
    return 0


class MyTestCase(unittest.TestCase):

    def build_networkx_graph(self):
        import networkx as nx

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

        return g

    def test_networkx_graph_dijkstra(self):
        # ------ SETUP THE TEST ------

        g = self.build_networkx_graph()

        def networkx_neighbor_function(node: str) -> Iterable[Tuple[float, str]]:
            # returns an iterable of tuples (edge_cost, node), just like the A* implementation wants
            return ((data['weight'], neighbor) for neighbor, data in g[node].items())

        # the node type in this case is plain string
        a_star: AStar[str] = AStar(neighbor_function=networkx_neighbor_function)

        # ------ RUN THE CODE ------

        # dummy heuristic turns A* into Dijkstra
        # run Dijkstra
        value, path, debug_out = a_star.run_with_debug(start='Start', end='Goal', heuristic_function=dummy_heuristic)

        # ------ ASSERT RESULTS CORRECT ------

        # check that the value is 14
        self.assertEqual(14, value)

        # check that the expected path is returned
        self.assertEqual(['Start', 'A', 'C', 'Goal'], path)

        # check that the debug output says that the items are opened in the exact order that we expect Dijkstra to do
        self.assertEqual(
            """Opening Start in distance 0, with predecessor Start.
Opening B in distance 1, with predecessor Start.
Opening H in distance 2, with predecessor B.
Opening A in distance 3, with predecessor Start.
Opening D in distance 4, with predecessor Start.
Opening O in distance 4, with predecessor H.
Opening C in distance 5, with predecessor A.
Opening F in distance 6, with predecessor C.
Opening G in distance 6, with predecessor H.
Opening L in distance 6, with predecessor D.
Opening E in distance 7, with predecessor B.
Opening I in distance 8, with predecessor H.
Opening K in distance 9, with predecessor G.
Opening N in distance 10, with predecessor K.
Opening J in distance 11, with predecessor E.
Opening M in distance 12, with predecessor N.
Opening Goal in distance 14, with predecessor C.
""", debug_out)

    def assert_path_correct(self, neighbor_function, path: list):
        for node, neighbor_actual in zip(path[:-1], path[1:]):
            neighbors_expected = [n for _, n in neighbor_function(node)]
            self.assertIn(neighbor_actual, neighbors_expected)

    def test_dynamically_generated_1d_grid(self):
        # ------ SETUP THE TEST ------

        NodeType = int

        def neighbor_function(node: NodeType) -> Iterable[Tuple[float, NodeType]]:
            for a in [-1, 1]:
                neighbor = node + a
                edge_cost = 1.

                yield edge_cost, neighbor

        a_star: AStar[NodeType] = AStar(neighbor_function=neighbor_function)

        start_node: NodeType = 0
        end_node: NodeType = 10

        def distance(node: NodeType, end: NodeType) -> float:
            return abs(node - end)

        # ------ RUN THE CODE ------

        dijkstra_value, dijkstra_path, dijkstra_debug = a_star.run_with_debug(start=start_node, end=end_node,
                                                                              heuristic_function=dummy_heuristic)

        astar_value, astar_path, astar_debug = a_star.run_with_debug(start=start_node, end=end_node,
                                                                     heuristic_function=distance)

        # ------ ASSERT RESULTS CORRECT ------

        # check value
        expected_value = distance(end_node, start_node)
        self.assertEqual(expected_value, dijkstra_value)
        self.assertEqual(expected_value, astar_value)

        # check path
        self.assert_path_correct(neighbor_function, dijkstra_path)
        self.assert_path_correct(neighbor_function, astar_path)

        # count the number of lines in the debug output
        dijkstra_n_openings = dijkstra_debug.count("Opening")
        astar_n_openings = astar_debug.count("Opening")

        # check that Dijkstra opens exactly 1 + 10 + 10 nodes (i.e. all in the range [-10, +10])
        self.assertEqual(21, dijkstra_n_openings)

        # check that A* opens exactly 1 + 10 nodes (i.e. only those in the range [0, +10])
        self.assertEqual(11, astar_n_openings)

        print("Dijkstra")
        print(dijkstra_debug)
        print()

        print("A*")
        print(astar_debug)
        print()

    def test_dynamically_generated_2d_grid(self):
        # ------ SETUP THE TEST ------

        NodeType = Tuple[int, int]

        def infinite_grid_neighbor_function(node: NodeType) -> Iterable[Tuple[float, NodeType]]:
            x, y = node
            for a, b in [[-1, 0], [0, -1], [1, 0], [0, 1]]:
                neighbor = x + a, y + b
                edge_cost = 1.

                yield edge_cost, neighbor

        a_star: AStar[NodeType] = AStar(neighbor_function=infinite_grid_neighbor_function)

        start_node: NodeType = (0, 0)
        end_node: NodeType = (10, 10)

        def euclidean_distance(node: NodeType, end: NodeType) -> float:
            xa, ya = node
            xb, yb = end

            return math.sqrt((xa - xb) ** 2 + (ya - yb) ** 2)

        def manhattan_distance(node: NodeType, end: NodeType) -> float:
            xa, ya = node
            xb, yb = end

            return abs(xa - xb) + abs(ya - yb)

        # ------ RUN THE CODE ------

        # run Dijkstra (dummy heuristic)
        dijkstra_value, dijkstra_path, dijkstra_debug = a_star.run_with_debug(start=start_node, end=end_node,
                                                                              heuristic_function=dummy_heuristic)

        # run A* with Euclidean distance heuristic
        euclidean_value, euclidean_path, euclidean_debug = a_star.run_with_debug(start=start_node, end=end_node,
                                                                                 heuristic_function=euclidean_distance)
        # run A* with Manhattan distance heuristic
        manhattan_value, manhattan_path, manhattan_debug = a_star.run_with_debug(start=start_node, end=end_node,
                                                                                 heuristic_function=manhattan_distance)

        # ------ ASSERT RESULTS CORRECT ------

        # check value correct
        expected_value = manhattan_distance(start_node, end_node)
        self.assertEqual(expected_value, dijkstra_value)
        self.assertEqual(expected_value, euclidean_value)
        self.assertEqual(expected_value, manhattan_value)

        # check path correct
        self.assert_path_correct(infinite_grid_neighbor_function, dijkstra_path)
        self.assert_path_correct(infinite_grid_neighbor_function, euclidean_path)
        self.assert_path_correct(infinite_grid_neighbor_function, manhattan_path)

        # count the number of lines in the debug output
        dijkstra_n_openings = dijkstra_debug.count("Opening")
        euclidean_n_openings = euclidean_debug.count("Opening")
        manhattan_n_openings = manhattan_debug.count("Opening")

        # check euclidean strictly better than dijkstra
        self.assertLess(euclidean_n_openings, dijkstra_n_openings)

        # check manhattan strictly better than euclidean
        self.assertLess(manhattan_n_openings, euclidean_n_openings)

        print("Dijkstra")
        print(dijkstra_debug)
        print()

        print("Euclidean")
        print(euclidean_debug)
        print()

        print("Manhattan")
        print(manhattan_debug)
        print()


if __name__ == '__main__':
    unittest.main()
