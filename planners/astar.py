"""
A* path planner.

Uses Manhattan distance as the heuristic since the Micromouse
moves only in the four cardinal directions.
"""

from heapq import heappop, heappush


class AStar:
    """A* search planner."""

    def __init__(self, graph):
        self.graph = graph

    # =====================================================
    # Heuristic
    # =====================================================

    @staticmethod
    def heuristic(
        a: tuple[int, int],
        b: tuple[int, int],
    ) -> int:
        """
        Manhattan distance.
        """

        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # =====================================================
    # Path reconstruction
    # =====================================================

    @staticmethod
    def reconstruct_path(
        came_from: dict,
        goal: tuple[int, int],
    ) -> list[tuple[int, int]]:
        """
        Reconstruct path from goal back to start.
        """

        path = [goal]

        current = goal

        while current in came_from:

            current = came_from[current]
            path.append(current)

        path.reverse()

        return path

    # =====================================================
    # Search
    # =====================================================

    def search(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> list[tuple[int, int]] | None:
        """
        Compute the shortest path from start to goal.

        Returns
        -------
        list[(row, col)] | None
        """

        frontier = []

        heappush(
            frontier,
            (
                self.heuristic(start, goal),
                0,          # g-cost
                start,
            ),
        )

        came_from = {}

        g_cost = {
            start: 0
        }

        visited = set()

        while frontier:

            _, current_cost, current = heappop(frontier)

            if current in visited:
                continue

            visited.add(current)

            if current == goal:

                return self.reconstruct_path(
                    came_from,
                    goal,
                )

            for neighbour in self.graph.get_neighbors(*current):

                if neighbour in visited:
                    continue

                tentative_cost = current_cost + 1

                if (
                    neighbour not in g_cost
                    or tentative_cost < g_cost[neighbour]
                ):

                    g_cost[neighbour] = tentative_cost

                    priority = (
                        tentative_cost
                        + self.heuristic(
                            neighbour,
                            goal,
                        )
                    )

                    came_from[neighbour] = current

                    heappush(
                        frontier,
                        (
                            priority,
                            tentative_cost,
                            neighbour,
                        ),
                    )

        return None