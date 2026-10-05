"""
Camera utilities for the MuJoCo viewer.

Adjust the constants below until you get the desired initial view.
The camera is NOT locked to the robot.
"""

import mujoco.viewer


# ==========================================================
# Camera Parameters (Tune these)
# ==========================================================

# World position that the camera looks at
LOOKAT_X = -3.3
LOOKAT_Y = 3.5
LOOKAT_Z = 0.0

# Zoom (larger = farther away)
CAMERA_DISTANCE = 5.0

# Rotation around the vertical axis (degrees)
CAMERA_AZIMUTH = 0

# Vertical viewing angle (degrees)
# -90 = straight down
# -45 = angled view
CAMERA_ELEVATION = -55.0


# ==========================================================
# Camera Setup
# ==========================================================

def setup_camera(viewer: mujoco.viewer.Handle) -> None:
    """
    Configure the initial camera.

    The camera is fixed at startup only.
    After the simulation starts, it behaves like the normal
    MuJoCo free camera.
    """

    viewer.cam.lookat[0] = LOOKAT_X
    viewer.cam.lookat[1] = LOOKAT_Y
    viewer.cam.lookat[2] = LOOKAT_Z

    viewer.cam.distance = CAMERA_DISTANCE
    viewer.cam.azimuth = CAMERA_AZIMUTH
    viewer.cam.elevation = CAMERA_ELEVATION

    viewer.sync()