"""
Graph representation of the maze.

Converts a Maze object into a graph that can be searched by
A*, Dijkstra, BFS, Flood Fill, or any graph-search algorithm.
"""

from environments.maze import Maze


class MazeGraph:
    """Graph interface for the maze."""

    # Row, Column offsets
    DIRECTIONS = (
        (-1, 0),   # North
        (1, 0),    # South
        (0, -1),   # West
        (0, 1),    # East
    )

    def __init__(self, maze: Maze):
        self.maze = maze

    # =====================================================
    # Bounds
    # =====================================================

    def in_bounds(
        self,
        row: int,
        col: int,
    ) -> bool:
        """
        Return True if the cell is inside the maze.
        """

        return (
            0 <= row < self.maze.rows
            and
            0 <= col < self.maze.cols
        )

    # =====================================================
    # Connectivity
    # =====================================================

    def is_connected(
        self,
        current: tuple[int, int],
        neighbour: tuple[int, int],
    ) -> bool:
        """
        Return True if two neighbouring cells are connected.
        """

        row, col = current
        nr, nc = neighbour

        dr = nr - row
        dc = nc - col

        wall_row = 2 * row + 1 + dr
        wall_col = 2 * col + 1 + dc

        return self.maze.grid[wall_row, wall_col] == 0

    # =====================================================
    # Neighbours
    # =====================================================

    def get_neighbors(
        self,
        row: int,
        col: int,
    ) -> list[tuple[int, int]]:
        """
        Return every reachable neighbouring cell.
        """

        neighbours = []

        for dr, dc in self.DIRECTIONS:

            nr = row + dr
            nc = col + dc

            if not self.in_bounds(nr, nc):
                continue

            if self.is_connected(
                (row, col),
                (nr, nc),
            ):
                neighbours.append((nr, nc))

        return neighbours

    # =====================================================
    # Utility
    # =====================================================

    def __contains__(
        self,
        cell: tuple[int, int],
    ) -> bool:

        return self.in_bounds(*cell)

    def __len__(self) -> int:
        return self.maze.rows * self.maze.cols