import time

import mujoco
import mujoco.viewer

from controllers.sensors import Sensors
from utils.camera import setup_camera

MODEL_PATH = "models/fixed_maze.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

sensors = Sensors(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:

    setup_camera(viewer)

    # Let the simulation settle.
    for _ in range(500):

        mujoco.mj_step(model, data)

        viewer.sync()

        time.sleep(model.opt.timestep)

    print()
    print("=== LIVE SENSOR TEST ===")
    print("Move/inspect the robot in the viewer.")
    print("Close the viewer window to exit.")
    print()

    while viewer.is_running():

        mujoco.mj_step(model, data)

        values = sensors.read()

        print(
            f"\r"
            f"Front: {values['front']:.3f} m   "
            f"Left: {values['left']:.3f} m   "
            f"Right: {values['right']:.3f} m",
            end="",
            flush=True
        )

        viewer.sync()

        time.sleep(0.05)

print("\nSensor test finished.")