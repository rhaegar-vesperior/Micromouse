"""
Low-level motion controller for the Micromouse.

Responsibilities
----------------
- Differential drive motor control
- Basic forward/reverse motion
- In-place turning
- Motion helper utilities

Higher-level navigation is handled by Navigator.
"""

import math
import time

import mujoco

from config import (
    CELL_SIZE,
    FORWARD_SPEED,
    TURN_SPEED,
    MAX_MOTOR_SPEED,
    TURN_TOLERANCE,
)


class Micromouse:
    """Low-level controller for the robot."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ):

        self.model = model
        self.data = data

        self.left_motor = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            "left_motor",
        )

        self.right_motor = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            "right_motor",
        )

        # Logical robot heading (East = 0 rad)
        # This is used as the reference for all future turns,
        # preventing accumulated yaw drift.
        self.target_yaw = 0.0

    # =====================================================
    # Motor Control
    # =====================================================

    def drive(
        self,
        left_speed: float,
        right_speed: float,
    ) -> None:
        """
        Drive left and right wheels.

        No speed clamping is performed. The requested motor
        commands are written directly to the actuators.

        Note:
            If the values exceed the actuator ctrlrange
            specified in the MuJoCo XML, MuJoCo will still
            limit them internally.
        """

        self.data.ctrl[self.left_motor] = left_speed
        self.data.ctrl[self.right_motor] = right_speed

    def stop(self) -> None:
        """Stop both wheels."""

        self.drive(0.0, 0.0)

    # =====================================================
    # Basic Motion
    # =====================================================

    def forward(
        self,
        speed: float = FORWARD_SPEED,
    ) -> None:

        self.drive(speed, speed)

    def reverse(
        self,
        speed: float = FORWARD_SPEED,
    ) -> None:

        self.drive(-speed, -speed)

    def turn_left(
        self,
        speed: float = TURN_SPEED,
    ) -> None:

        self.drive(speed, -speed)

    def turn_right(
        self,
        speed: float = TURN_SPEED,
    ) -> None:

        self.drive(-speed, speed)

    # =====================================================
    # Simulation Step
    # =====================================================

    def _step(self, viewer) -> None:
        """
        Advance simulation by one timestep.
        """

        mujoco.mj_step(
            self.model,
            self.data,
        )

        viewer.sync()

        time.sleep(
            self.model.opt.timestep
        )

    # =====================================================
    # Geometry Helpers
    # =====================================================

    @staticmethod
    def normalize(angle: float) -> float:
        """
        Normalize an angle to [-π, π].
        """

        while angle > math.pi:
            angle -= 2 * math.pi

        while angle < -math.pi:
            angle += 2 * math.pi

        return angle

    @staticmethod
    def distance(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> float:
        """
        Euclidean distance.
        """

        return math.hypot(
            x2 - x1,
            y2 - y1,
        )




    # =====================================================
    # Drive a specified distance
    # =====================================================

    def forward_distance(
        self,
        localization,
        viewer,
        distance: float,
        speed: float = FORWARD_SPEED,
    ) -> None:
        """
        Drive forward for a specified distance.

        Used only for entering the maze before cell-based navigation
        begins.
        """

        start_x, start_y, _ = localization.pose()

        self.forward(speed)

        while viewer.is_running():

            self._step(viewer)

            x, y, _ = localization.pose()

            travelled = self.distance(
                start_x,
                start_y,
                x,
                y,
            )

            if travelled >= distance:
                break

        self.stop()

    # =====================================================
    # Move exactly one maze cell
    # =====================================================

    def move_one_cell(
        self,
        localization,
        sensors,
        viewer,
        speed: float = FORWARD_SPEED,
    ) -> bool:
        """
        Move exactly one maze cell.

        This version performs pure distance-based motion using
        localization only. Sensor-based wall following has been
        disabled so that navigation relies entirely on the A*
        planner and heading control.
        """

        start_x, start_y, _ = localization.pose()

        # Drive straight
        self.forward(speed)

        while viewer.is_running():

            self._step(viewer)

            x, y, _ = localization.pose()

            travelled = self.distance(
                start_x,
                start_y,
                x,
                y,
            )

            if travelled >= CELL_SIZE:

                self.stop()

                return True

        self.stop()

        return False


    # =====================================================
    # Turn left by 90 degrees
    # =====================================================

    def turn_left_90(
        self,
        localization,
        viewer,
    ) -> None:
        """
        Rotate the robot exactly 90° counter-clockwise.

        The desired heading is computed from the robot's
        logical heading rather than the measured yaw.
        This prevents accumulated turning error.
        """

        # Update logical heading
        self.target_yaw = self.normalize(
            self.target_yaw + math.pi / 2
        )

        # ------------------------------
        # Phase 1 : Fast turn
        # ------------------------------

        while viewer.is_running():

            _, _, current_yaw = localization.pose()

            error = self.normalize(
                self.target_yaw - current_yaw
            )

            if abs(error) <= math.radians(10):
                break

            if abs(error) > math.radians(35):
                speed = TURN_SPEED
            else:
                speed = max(
                    0.30,
                    TURN_SPEED * 0.45,
                )

            self.turn_left(speed)

            self._step(viewer)

        self.stop()

        # Allow physics to settle
        for _ in range(5):
            self._step(viewer)

        # ------------------------------
        # Phase 2 : Fine alignment
        # ------------------------------

        while viewer.is_running():

            _, _, current_yaw = localization.pose()

            error = self.normalize(
                self.target_yaw - current_yaw
            )

            if abs(error) <= TURN_TOLERANCE:
                break

            correction_speed = 0.40

            if error > 0:
                self.turn_left(correction_speed)
            else:
                self.turn_right(correction_speed)

            self._step(viewer)

        self.stop()

        # Final settling
        for _ in range(3):
            self._step(viewer)

    # =====================================================
    # Turn right by 90 degrees
    # =====================================================

    def turn_right_90(
        self,
        localization,
        viewer,
    ) -> None:
        """
        Rotate the robot exactly 90° clockwise.

        The desired heading is computed from the robot's
        logical heading rather than the measured yaw.
        This prevents accumulated turning error.
        """

        # Update logical heading
        self.target_yaw = self.normalize(
            self.target_yaw - math.pi / 2
        )

        # ------------------------------
        # Phase 1 : Fast turn
        # ------------------------------

        while viewer.is_running():

            _, _, current_yaw = localization.pose()

            error = self.normalize(
                self.target_yaw - current_yaw
            )

            if abs(error) <= math.radians(10):
                break

            if abs(error) > math.radians(35):
                speed = TURN_SPEED
            else:
                speed = max(
                    0.30,
                    TURN_SPEED * 0.45,
                )

            self.turn_right(speed)

            self._step(viewer)

        self.stop()

        # Allow physics to settle
        for _ in range(5):
            self._step(viewer)

        # ------------------------------
        # Phase 2 : Fine alignment
        # ------------------------------

        while viewer.is_running():

            _, _, current_yaw = localization.pose()

            error = self.normalize(
                self.target_yaw - current_yaw
            )

            if abs(error) <= TURN_TOLERANCE:
                break

            correction_speed = 0.40

            if error < 0:
                self.turn_right(correction_speed)
            else:
                self.turn_left(correction_speed)

            self._step(viewer)

        self.stop()

        # Final settling
        for _ in range(3):
            self._step(viewer)

    # =====================================================
    # Turn around (180 degrees)
    # =====================================================

    def turn_around(
        self,
        localization,
        viewer,
    ) -> None:
        """
        Rotate the robot by 180°.
        """

        self.turn_right_90(localization, viewer)
        self.turn_right_90(localization, viewer)