import numpy as np


class Maze:

    def __init__(self, rows=16, cols=16, seed=42):

        self.rows = rows
        self.cols = cols
        self.seed = seed

        self.grid = np.ones(
            (rows * 2 + 1, cols * 2 + 1),
            dtype=np.int8
        )

        self.start = (0, 0)
        self.exit = (rows - 1, cols - 1)

        self._generate()

    # =====================================================
    # GENERATE MAZE
    # =====================================================

    def _generate(self):

        rng = np.random.default_rng(self.seed)

        visited = np.zeros(
            (self.rows, self.cols),
            dtype=bool
        )

        stack = [self.start]
        visited[self.start] = True

        # Open starting cell

        r, c = self.start

        self.grid[
            2 * r + 1,
            2 * c + 1
        ] = 0

        directions = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
        ]

        # -------------------------------------------------
        # Randomized DFS
        # -------------------------------------------------

        while stack:

            r, c = stack[-1]

            neighbors = []

            for dr, dc in directions:

                nr = r + dr
                nc = c + dc

                if (
                    0 <= nr < self.rows
                    and 0 <= nc < self.cols
                    and not visited[nr, nc]
                ):
                    neighbors.append((nr, nc))

            if not neighbors:

                stack.pop()
                continue

            nr, nc = neighbors[
                rng.integers(len(neighbors))
            ]

            # Remove wall between cells

            wall_r = 2 * r + 1 + (nr - r)
            wall_c = 2 * c + 1 + (nc - c)

            self.grid[wall_r, wall_c] = 0

            self.grid[
                2 * nr + 1,
                2 * nc + 1
            ] = 0

            visited[nr, nc] = True

            stack.append((nr, nc))

        # -------------------------------------------------
        # Braid the maze
        #
        # Remove selected dead-end walls.
        # This creates alternative routes.
        # -------------------------------------------------

        dead_end_cells = []

        for r in range(self.rows):

            for c in range(self.cols):

                if (r, c) in (
                    self.start,
                    self.exit
                ):
                    continue

                open_count = 0

                for dr, dc in directions:

                    nr = r + dr
                    nc = c + dc

                    if not (
                        0 <= nr < self.rows
                        and 0 <= nc < self.cols
                    ):
                        continue

                    wall_r = 2 * r + 1 + dr
                    wall_c = 2 * c + 1 + dc

                    if self.grid[wall_r, wall_c] == 0:
                        open_count += 1

                if open_count == 1:

                    dead_end_cells.append(
                        (r, c)
                    )

        rng.shuffle(dead_end_cells)

        # Remove roughly 35% of dead ends.

        for r, c in dead_end_cells:

            if rng.random() > 0.35:
                continue

            candidates = []

            for dr, dc in directions:

                nr = r + dr
                nc = c + dc

                if not (
                    0 <= nr < self.rows
                    and 0 <= nc < self.cols
                ):
                    continue

                wall_r = 2 * r + 1 + dr
                wall_c = 2 * c + 1 + dc

                if self.grid[wall_r, wall_c] == 1:

                    candidates.append(
                        (wall_r, wall_c)
                    )

            if candidates:

                wall_r, wall_c = candidates[
                    rng.integers(len(candidates))
                ]

                self.grid[
                    wall_r,
                    wall_c
                ] = 0

        # -------------------------------------------------
        # Entrance
        # -------------------------------------------------

        self.grid[1, 0] = 0

        # -------------------------------------------------
        # Exit
        # -------------------------------------------------

        self.grid[
            2 * self.rows - 1,
            2 * self.cols
        ] = 0

    # =====================================================
    # PRINT
    # =====================================================

    def print(self):

        for row in self.grid:

            print(
                "".join(
                    "█" if x else " "
                    for x in row
                )
            )