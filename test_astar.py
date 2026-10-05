from environments.maze import Maze
from planners.graph import MazeGraph
from planners.astar import AStar

maze = Maze(seed=42)

graph = MazeGraph(maze)

planner = AStar(graph)

path = planner.search(
    maze.start,
    maze.exit
)

print(path)
print()

print("Length:", len(path))