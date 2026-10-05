import math
import time

import mujoco
import mujoco.viewer

from controllers.micromouse import Micromouse
from controllers.localization import Localization
from utils.camera import setup_camera

MODEL_PATH = "models/fixed_maze.xml"

TURN_SPEED = 0.5

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

robot = Micromouse(model, data)
loc = Localization(model, data)


def normalize(angle):
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


with mujoco.viewer.launch_passive(model, data) as viewer:

    setup_camera(viewer)

    # settle
    for _ in range(300):
        mujoco.mj_step(model, data)

    _, _, start_yaw = loc.pose()

    print(f"Start yaw = {math.degrees(start_yaw):.1f}")

    robot.turn_left(TURN_SPEED)

    while viewer.is_running():

        mujoco.mj_step(model, data)

        _, _, yaw = loc.pose()

        diff = normalize(yaw - start_yaw)

        print(
            f"Turned = {math.degrees(diff):6.1f}",
            end="\r"
        )

        if abs(diff) >= math.pi / 2:

            robot.stop()

            print()

            print(
                f"Final yaw = {math.degrees(yaw):.1f}"
            )

            break

        viewer.sync()
        time.sleep(model.opt.timestep)

    while viewer.is_running():
        viewer.sync()
        time.sleep(0.01)