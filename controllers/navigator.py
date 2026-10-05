"""
High-level navigation controller.

Responsibilities
----------------
- Maintain robot heading
- Rotate to required direction
- Execute an A* path cell-by-cell
"""

from enum import Enum


class Heading(Enum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3


class Navigator:

    def __init__(
        self,
        robot,
        localization,
        sensors,
        viewer,
    ):
        self.robot = robot
        self.localization = localization
        self.sensors = sensors
        self.viewer = viewer

        # Robot enters maze from the west,
        # therefore initially faces east.
        self.heading = Heading.EAST

    # =====================================================
    # Heading between two neighbouring cells
    # =====================================================

    @staticmethod
    def heading_between(current, nxt):
        """
        Determine heading required to move from
        current cell to neighbouring cell.
        """

        r1, c1 = current
        r2, c2 = nxt

        if r2 == r1 - 1:
            return Heading.NORTH

        if r2 == r1 + 1:
            return Heading.SOUTH

        if c2 == c1 + 1:
            return Heading.EAST

        if c2 == c1 - 1:
            return Heading.WEST

        raise ValueError(
            f"Cells {current} and {nxt} are not adjacent."
        )

    # =====================================================
    # Rotate robot
    # =====================================================

    def rotate_to(self, target):
        """
        Rotate robot until it faces the target heading.
        """

        diff = (
            target.value - self.heading.value
        ) % 4

        if diff == 0:
            return

        elif diff == 1:

            self.robot.turn_right_90(
                self.localization,
                self.viewer,
            )

        elif diff == 3:

            self.robot.turn_left_90(
                self.localization,
                self.viewer,
            )

        elif diff == 2:

            self.robot.turn_around(
                self.localization,
                self.viewer,
            )

        self.heading = target

    # =====================================================
    # Execute complete path
    # =====================================================

    def follow_path(self, path):
        """
        Execute an A* path.

        Parameters
        ----------
        path : list[(row, col)]

        Returns
        -------
        bool
        """

        if path is None or len(path) < 2:
            print("No path to execute.")
            return False

        print("\n========== EXECUTING PATH ==========\n")

        for current, nxt in zip(path[:-1], path[1:]):

            print(f"{current} -> {nxt}")

            desired_heading = self.heading_between(
                current,
                nxt,
            )

            self.rotate_to(desired_heading)

            success = self.robot.move_one_cell(
                localization=self.localization,
                sensors=self.sensors,
                viewer=self.viewer,
            )

            if not success:

                self.robot.stop()

                print(
                    f"Navigation failed while moving "
                    f"from {current} to {nxt}."
                )

                return False

        self.robot.stop()

        print("\nPath completed successfully.")

        return True