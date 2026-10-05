import time

import mujoco
import mujoco.viewer

from controllers.micromouse import Micromouse
from controllers.localization import Localization
from utils.camera import setup_camera

MODEL_PATH = "models/fixed_maze.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

robot = Micromouse(model, data)
loc = Localization(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:

    setup_camera(viewer)

    print("\n===== LOCALIZATION TEST =====\n")

    while viewer.is_running():

        # Drive robot forward
        robot.forward(0.5)

        # Advance simulation
        mujoco.mj_step(model, data)

        # Read pose
        x, y, yaw = loc.pose()

        print(
            f"x = {x:.3f}    "
            f"y = {y:.3f}    "
            f"yaw = {yaw:.3f}"
        )

        viewer.sync()

        time.sleep(model.opt.timestep)