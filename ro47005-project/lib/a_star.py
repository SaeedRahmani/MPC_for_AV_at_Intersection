from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Callable, Iterable, List, TypeVar, Hashable, Tuple, Dict, Generic

# generic node type -> can be any type in practice; this is just for IDE autocompletion hints
TNode = TypeVar("TNode", bound=Hashable)


@dataclass
class AStarDebugData(Generic[TNode]):
    g: float
    h: float
    node: TNode
    predecessor: TNode


class AStar(Generic[TNode]):

    def __init__(self, neighbor_function: Callable[[TNode], Iterable[Tuple[float, TNode]]]):
        self.neighbor_function = neighbor_function
        self._debug_data: List[AStarDebugData[TNode]] = []

    @property
    def debug_data(self):
        """
        Read-only debug data property (accessible if debug=True on run() function)
        :return:
        """
        return self._debug_data

    def run(self, start: TNode, is_goal_function: Callable[[TNode], bool],
            heuristic_function: Callable[[TNode], float],
            debug=False) -> Tuple[float, List[TNode]]:
        q: List[Tuple[float, float, TNode, TNode]] = [(0, 0, start, start)]  # G + H value, G value, node, predecessor

        if debug:
            self._debug_data = []

        # best predecessor dict
        pred_dict: Dict[TNode, Tuple[float, TNode]] = {}

        while q:
            gh, g, node, predecessor = heappop(q)

            if node in pred_dict and g >= pred_dict[node][0]:
                # we have seen this node before -> skip
                # TODO: if our heuristic is consistent,
                #  we can remove `and g >= pred_dict[node][0]`
                continue

            if debug:
                self._debug_data.append(AStarDebugData(g=g, h=gh - g, node=node, predecessor=predecessor))

            # store value predecessor
            pred_dict[node] = g, predecessor

            if is_goal_function(node):
                # we are done
                # reconstruct path
                path = [node]

                while node != start:
                    path.append(predecessor)
                    node, predecessor = predecessor, pred_dict[predecessor][1]

                path.reverse()
                return g, path

            # process neighbors
            for edge_value, neighbor in self.neighbor_function(node):
                neighbor_g = g + edge_value

                if neighbor not in pred_dict or neighbor_g < pred_dict[neighbor][0]:
                    # this is a better path to the node in question than we had before
                    neighbor_gh = neighbor_g + heuristic_function(neighbor)
                    heappush(q, (neighbor_gh, neighbor_g, neighbor, node))

        raise Exception("No solution found.")
