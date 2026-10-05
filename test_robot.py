import time

import mujoco
import mujoco.viewer

from utils.camera import setup_camera

MODEL_PATH = "models/micromouse.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# Compute initial kinematics
mujoco.mj_forward(model, data)

robot_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "robot"
)

left_wheel_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "left_wheel"
)

right_wheel_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "right_wheel"
)

left_geom_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_GEOM,
    "left_wheel_geom"
)

right_geom_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_GEOM,
    "right_wheel_geom"
)

print("=== MODEL ===")
print("Robot body:", robot_id)
print("Left wheel body:", left_wheel_id)
print("Right wheel body:", right_wheel_id)

print("\n=== INITIAL POSITIONS ===")
print("Robot:", data.xpos[robot_id])
print("Left wheel:", data.xpos[left_wheel_id])
print("Right wheel:", data.xpos[right_wheel_id])

print("\n=== WHEEL GEOMETRY ORIENTATION ===")
print("Left wheel:")
print(data.geom_xmat[left_geom_id].reshape(3, 3))

print("\nRight wheel:")
print(data.geom_xmat[right_geom_id].reshape(3, 3))

# Drive both wheels forward
data.ctrl[0] = 1.0
data.ctrl[1] = 1.0

with mujoco.viewer.launch_passive(model, data) as viewer:

    setup_camera(viewer)

    for step in range(5000):

        if not viewer.is_running():
            break

        mujoco.mj_step(model, data)

        viewer.sync()

        if step % 500 == 0:

            print(f"\n--- STEP {step} ---")

            print("Robot position:")
            print(data.qpos[:3])

            print("Robot X:", data.qpos[0])
            print("Robot Y:", data.qpos[1])
            print("Robot yaw:", data.qpos[2])

            print("Robot X velocity:", data.qvel[0])
            print("Robot Y velocity:", data.qvel[1])
            print("Robot yaw velocity:", data.qvel[2])

            print("Left wheel velocity:", data.qvel[3])
            print("Right wheel velocity:", data.qvel[4])

            print("Contacts:", data.ncon)

        time.sleep(model.opt.timestep)

    print("\nTest complete.")

    while viewer.is_running():

        viewer.sync()

        time.sleep(0.01)