import time

import mujoco
import mujoco.viewer

from utils.camera import setup_camera

MODEL_PATH = "models/fixed_maze.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

robot_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "robot"
)

with mujoco.viewer.launch_passive(
    model,
    data,
) as viewer:

    setup_camera(viewer)

    # Update all derived quantities.
    mujoco.mj_forward(model, data)

    print("=== ROBOT BODY ===")
    print("Body ID:", robot_id)

    print("\n=== INITIAL WORLD POSE ===")

    print(
        "World position:",
        data.xpos[robot_id]
    )

    print(
        "World orientation matrix:"
    )

    print(
        data.xmat[robot_id].reshape(3, 3)
    )

    print("\n=== JOINT COORDINATES ===")

    print(
        "qpos:",
        data.qpos[:]
    )

    print("\nClose the viewer window to exit.")

    while viewer.is_running():

        viewer.sync()

        time.sleep(0.01)

print("Viewer closed.")