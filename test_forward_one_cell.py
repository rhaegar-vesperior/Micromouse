import math
import time

import mujoco
import mujoco.viewer

from controllers.micromouse import Micromouse
from controllers.localization import Localization
from utils.camera import setup_camera

MODEL_PATH = "models/fixed_maze.xml"

CELL_SIZE = 0.60
SPEED = 0.5

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

robot = Micromouse(model, data)
loc = Localization(model, data)


def distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)


with mujoco.viewer.launch_passive(model, data) as viewer:

    setup_camera(viewer)

    print("\n=== FORWARD ONE CELL TEST ===\n")

    # settle physics
    for _ in range(300):
        mujoco.mj_step(model, data)

    start_x, start_y, _ = loc.pose()

    print(f"Start : ({start_x:.3f}, {start_y:.3f})")

    robot.forward(SPEED)

    while viewer.is_running():

        mujoco.mj_step(model, data)

        x, y, yaw = loc.pose()

        d = distance(start_x, start_y, x, y)

        print(
            f"distance = {d:.3f}",
            end="\r"
        )

        if d >= CELL_SIZE:

            robot.stop()

            print()

            print("Reached one cell.")

            print(f"End : ({x:.3f}, {y:.3f})")

            break

        viewer.sync()

        time.sleep(model.opt.timestep)

    while viewer.is_running():

        viewer.sync()

        time.sleep(0.01)