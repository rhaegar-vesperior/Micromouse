from environments.maze import Maze
from planners.graph import MazeGraph
from planners.astar import AStar
from planners.planner import PathPlanner

maze = Maze(seed=42)

graph = MazeGraph(maze)

astar = AStar(graph)

path = astar.search(
    maze.start,
    maze.exit
)

planner = PathPlanner()

actions = planner.actions_from_path(path)

print("PATH")
print(path)

print("\nACTIONS")
for action in actions:
    print(action)