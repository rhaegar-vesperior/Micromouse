# main.py

import time
import mujoco
import mujoco.viewer

from controllers.movement_controller import MovementController


MODEL_PATH = "models/micromouse.xml"


def run_for(controller, model, data, viewer, duration):
    """Run the simulation for the specified duration (seconds)."""
    start = time.time()

    while viewer.is_running() and (time.time() - start < duration):
        mujoco.mj_step(model, data)
        viewer.sync()


def main():
    # Load model
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    # Create controller
    controller = MovementController(model, data)

    # Launch viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:

        # Let everything settle
        run_for(controller, model, data, viewer, 1.0)

        print("Forward")
        controller.forward(2.0)
        run_for(controller, model, data, viewer, 2.0)

        print("Stop")
        controller.stop()
        run_for(controller, model, data, viewer, 1.0)

        print("Turn Left")
        controller.turn_left(1.5)
        run_for(controller, model, data, viewer, 1.5)

        print("Stop")
        controller.stop()
        run_for(controller, model, data, viewer, 1.0)

        print("Forward")
        controller.forward(2.0)
        run_for(controller, model, data, viewer, 2.0)

        print("Done")
        controller.stop()

        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()