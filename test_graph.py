from environments.maze import Maze
from planners.graph import MazeGraph

maze = Maze()

graph = MazeGraph(maze)

print(graph.get_neighbors(0, 0))
print(graph.get_neighbors(5, 5))
print(graph.get_neighbors(15, 15))