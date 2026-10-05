"""
Main entry point for the Micromouse simulation.

Pipeline
--------
1. Load MuJoCo model
2. Initialize controllers
3. Generate maze
4. Compute A* path
5. Launch simulator
6. Execute path
"""

import time

import mujoco
import mujoco.viewer

from config import SETTLE_STEPS

from controllers.localization import Localization
from controllers.micromouse import Micromouse
from controllers.navigator import Navigator
from controllers.sensors import Sensors

from environments.maze import Maze

from planners.graph import MazeGraph
from planners.astar import AStar

from utils.camera import setup_camera


MODEL_PATH = "models/fixed_maze.xml"


# ==========================================================
# Planning
# ==========================================================

def create_path():
    """Generate a maze and compute the shortest path."""

    maze = Maze()

    graph = MazeGraph(maze)

    planner = AStar(graph)

    path = planner.search(
        maze.start,
        maze.exit,
    )

    return maze, path


# ==========================================================
# Printing
# ==========================================================

def print_path(path):

    print("\n" + "=" * 60)
    print("A* SHORTEST PATH")
    print("=" * 60)

    for cell in path:
        print(cell)

    print("-" * 60)
    print(f"Cells : {len(path)}")
    print("=" * 60 + "\n")


# ==========================================================
# Main
# ==========================================================

def main():

    print("\nLoading simulation...")

    try:

        # --------------------------------------------------
        # Load model
        # --------------------------------------------------

        model = mujoco.MjModel.from_xml_path(
            MODEL_PATH
        )

        data = mujoco.MjData(model)

        # --------------------------------------------------
        # Controllers
        # --------------------------------------------------

        robot = Micromouse(
            model,
            data,
        )

        localization = Localization(
            model,
            data,
        )

        sensors = Sensors(
            model,
            data,
        )

        # --------------------------------------------------
        # Path Planning
        # --------------------------------------------------

        maze, path = create_path()

        if path is None:

            print("ERROR: No valid path found.")

            return

        print_path(path)

        # --------------------------------------------------
        # Launch Viewer
        # --------------------------------------------------

        with mujoco.viewer.launch_passive(
            model,
            data,
        ) as viewer:

            setup_camera(viewer)

            # ----------------------------------------------
            # Let physics settle
            # ----------------------------------------------

            robot.stop()

            print("Settling physics...")

            for _ in range(SETTLE_STEPS):

                mujoco.mj_step(model, data)

                viewer.sync()

                time.sleep(model.opt.timestep)

            # ----------------------------------------------
            # Synchronize logical heading with actual heading
            # ----------------------------------------------

            _, _, yaw = localization.pose()

            robot.target_yaw = robot.normalize(yaw)

            print(
                f"Initial heading: "
                f"{round(robot.target_yaw * 180 / 3.14159265, 2)}°"
            )

            # ----------------------------------------------
            # Navigator
            # ----------------------------------------------

            navigator = Navigator(
                robot=robot,
                localization=localization,
                sensors=sensors,
                viewer=viewer,
            )

            # ----------------------------------------------
            # Enter Maze
            # ----------------------------------------------

            print("\n" + "=" * 60)
            print("ENTERING MAZE")
            print("=" * 60)

            robot.forward_distance(
                localization=localization,
                viewer=viewer,
                distance=0.50,
            )

            # ----------------------------------------------
            # Execute Path
            # ----------------------------------------------

            print("\n" + "=" * 60)
            print("FOLLOWING PATH")
            print("=" * 60)

            success = navigator.follow_path(path)

            robot.stop()

            print()

            if success:

                print("=" * 60)
                print("GOAL REACHED")
                print("=" * 60)

            else:

                print("=" * 60)
                print("NAVIGATION FAILED")
                print("=" * 60)

            # ----------------------------------------------
            # Keep viewer alive
            # ----------------------------------------------

            while viewer.is_running():

                viewer.sync()

                time.sleep(0.01)

    except FileNotFoundError:

        print(f"Model not found: {MODEL_PATH}")

    except Exception as exc:

        print(f"\nSimulation error:\n{exc}")

    finally:

        try:
            robot.stop()
        except Exception:
            pass


# ==========================================================
# Entry
# ==========================================================

if __name__ == "__main__":
    main()