"""
Robot localization using MuJoCo ground-truth pose.

This module provides the robot position and heading directly from the
MuJoCo simulation. It is used by the navigation and motion controllers.
"""

import math

import mujoco


class Localization:
    """Provides the robot pose in world coordinates."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):

        self.data = data

        self.robot_body = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            "robot",
        )

    # =====================================================
    # Position
    # =====================================================

    @property
    def x(self) -> float:
        """Robot x-position (m)."""
        return float(self.data.xpos[self.robot_body][0])

    @property
    def y(self) -> float:
        """Robot y-position (m)."""
        return float(self.data.xpos[self.robot_body][1])

    # =====================================================
    # Orientation
    # =====================================================

    @property
    def yaw(self) -> float:
        """
        Robot heading in radians.

        Returns
        -------
        float
            Heading in the range [-π, π].
        """

        rotation = self.data.xmat[self.robot_body].reshape(3, 3)

        return math.atan2(
            rotation[1, 0],
            rotation[0, 0],
        )

    # =====================================================
    # Complete pose
    # =====================================================

    def pose(self) -> tuple[float, float, float]:
        """
        Returns
        -------
        tuple
            (x, y, yaw)
        """

        return (
            self.x,
            self.y,
            self.yaw,
        )

    # =====================================================
    # Utilities
    # =====================================================

    def position(self) -> tuple[float, float]:
        """Return only the robot position."""

        return (
            self.x,
            self.y,
        )

    # =====================================================
    # Debug
    # =====================================================

    def __str__(self) -> str:

        return (
            f"x={self.x:.3f} m | "
            f"y={self.y:.3f} m | "
            f"yaw={math.degrees(self.yaw):.1f}°"
        )