import time

import mujoco
import mujoco.viewer

from controllers.micromouse import Micromouse

MODEL_PATH = "models/fixed_maze.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

robot = Micromouse(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:

    print("LEFT MOTOR ONLY")

    robot.drive(1.0, 0.0)

    start = time.time()

    while viewer.is_running() and time.time() - start < 3:

        mujoco.mj_step(model, data)
        viewer.sync()

    robot.stop()

    time.sleep(2)

    print("RIGHT MOTOR ONLY")

    robot.drive(0.0, 1.0)

    start = time.time()

    while viewer.is_running() and time.time() - start < 3:

        mujoco.mj_step(model, data)
        viewer.sync()

    robot.stop()

    while viewer.is_running():

        viewer.sync()
        time.sleep(0.01)