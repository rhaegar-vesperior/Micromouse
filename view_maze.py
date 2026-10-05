import time

import mujoco
import mujoco.viewer
from utils.camera import setup_camera


MODEL_PATH = "models/fixed_maze.xml"


model = mujoco.MjModel.from_xml_path(
    MODEL_PATH
)

data = mujoco.MjData(model)


with mujoco.viewer.launch_passive(
    model,
    data,
) as viewer:

    setup_camera(viewer)

    while viewer.is_running():

        mujoco.mj_step(
            model,
            data
        )

        viewer.sync()

        time.sleep(
            model.opt.timestep
        )