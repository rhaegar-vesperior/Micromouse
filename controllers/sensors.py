"""
Sensor interface for the Micromouse.

Provides access to the MuJoCo rangefinder sensors and helper
functions for wall detection.
"""

from typing import Dict

import mujoco

class Sensors:
    """Wrapper around MuJoCo rangefinder sensors."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):

        self.model = model
        self.data = data

        self.sensor_ids = {
            "front": mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_SENSOR,
                "front_distance",
            ),
            "left": mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_SENSOR,
                "left_distance",
            ),
            "right": mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_SENSOR,
                "right_distance",
            ),
        }

    # ==========================================================
    # Individual readings
    # ==========================================================

    @property
    def front(self) -> float:
        return float(self.data.sensordata[self.sensor_ids["front"]])

    @property
    def left(self) -> float:
        return float(self.data.sensordata[self.sensor_ids["left"]])

    @property
    def right(self) -> float:
        return float(self.data.sensordata[self.sensor_ids["right"]])

    # ==========================================================
    # Read all
    # ==========================================================

    def read(self) -> Dict[str, float]:
        """Return all sensor readings."""

        return {
            "front": self.front,
            "left": self.left,
            "right": self.right,
        }

    # ==========================================================
    # Debug
    # ==========================================================

    def __str__(self) -> str:

        return (
            f"Front={self.front:.3f} m | "
            f"Left={self.left:.3f} m | "
            f"Right={self.right:.3f} m"
        )