from environments.maze import Maze
import numpy as np


def main():

    maze = Maze(
        rows=16,
        cols=16
    )

    maze.print()

    np.save(
        "environments/fixed_maze.npy",
        maze.grid
    )

    print()
    print("Maze saved.")
    print("Grid:", maze.grid.shape)
    print("Start:", maze.start)
    print("Exit:", maze.exit)


if __name__ == "__main__":
    main()