import time

import mujoco
import mujoco.viewer

from controllers.micromouse import Micromouse
from controllers.sensors import Sensors
from utils.camera import setup_camera

MODEL_PATH = "models/fixed_maze.xml"

TURN_DISTANCE = 0.20       # Half body length
FORWARD_SPEED = 1.0
TURN_SPEED = 0.8
TURN_TIME = 2.0           # Adjust if not exactly 90°

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

robot = Micromouse(model, data)
sensors = Sensors(model, data)

state = "FORWARD"

turn_start = 0.0

with mujoco.viewer.launch_passive(model, data) as viewer:

    setup_camera(viewer)

    print()
    print("===== FORWARD -> TURN TEST =====")
    print()

    while viewer.is_running():

        mujoco.mj_step(model, data)

        values = sensors.read()

        front = values["front"]

        if state == "FORWARD":

            robot.forward(FORWARD_SPEED)

            if front <= TURN_DISTANCE:

                print(
                    f"Wall detected at {front:.3f} m"
                )

                robot.drive(
                    0.0,
                    TURN_SPEED
                )

                turn_start = time.time()

                state = "TURN"

        elif state == "TURN":

            robot.drive(
                0.0,
                TURN_SPEED
            )

            if time.time() - turn_start >= TURN_TIME:

                print("Turn complete.")

                state = "FORWARD"

        viewer.sync()

        time.sleep(model.opt.timestep)