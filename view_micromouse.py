import mujoco
import mujoco.viewer


MODEL_PATH = "models/micromouse.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# Left and right wheel target velocities
data.ctrl[0] = 10.0
data.ctrl[1] = 10.0

# Start MuJoCo's managed viewer
mujoco.viewer.launch(model, data)