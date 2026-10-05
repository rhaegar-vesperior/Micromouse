import time

import mujoco
import mujoco.viewer

from controllers.micromouse import Micromouse
from utils.camera import setup_camera

MODEL_PATH = "models/fixed_maze.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

robot = Micromouse(model, data)

print("=== INITIAL ===")
print("X:", data.qpos[0])
print("Y:", data.qpos[1])
print("Yaw:", data.qpos[2])

with mujoco.viewer.launch_passive(model, data) as viewer:

    setup_camera(viewer)

    # Start stopped.
    robot.stop()

    # Let physics settle.
    for _ in range(500):

        mujoco.mj_step(model, data)

        viewer.sync()

        time.sleep(model.opt.timestep)

    start_x = data.qpos[0]
    start_y = data.qpos[1]
    start_yaw = data.qpos[2]

    print("\n=== START ===")
    print("X:", start_x)
    print("Y:", start_y)
    print("Yaw:", start_yaw)

    # -----------------------------------------------------
    # DRIVE FORWARD
    # -----------------------------------------------------

    robot.drive(1.0, 1.0)

    start_time = time.time()

    while (
        viewer.is_running()
        and time.time() - start_time < 1.0
    ):

        mujoco.mj_step(model, data)

        viewer.sync()

        time.sleep(model.opt.timestep)

    robot.stop()

    # -----------------------------------------------------
    # RESULTS
    # -----------------------------------------------------

    end_x = data.qpos[0]
    end_y = data.qpos[1]
    end_yaw = data.qpos[2]

    print("\n=== RESULT ===")
    print("X change:", end_x - start_x)
    print("Y change:", end_y - start_y)
    print("Yaw change:", end_yaw - start_yaw)

    print("\nFinal:")
    print("X:", end_x)
    print("Y:", end_y)
    print("Yaw:", end_yaw)

    # Keep viewer open
    while viewer.is_running():

        viewer.sync()

        time.sleep(0.01)