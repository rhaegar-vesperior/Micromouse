"""
Convert a cell path into executable robot actions.
"""

from enum import Enum, auto


class Heading(Enum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3


class Action(Enum):
    FORWARD = auto()
    LEFT = auto()
    RIGHT = auto()
    TURN_AROUND = auto()


class PathPlanner:
    """
    Converts a path into robot actions.
    """

    @staticmethod
    def heading_between(
        current: tuple[int, int],
        nxt: tuple[int, int],
    ) -> Heading:
        """
        Return the heading required to move from one cell
        to an adjacent cell.
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

    @staticmethod
    def turn_actions(
        current: Heading,
        target: Heading,
    ) -> list[Action]:
        """
        Return the actions required to rotate from the
        current heading to the target heading.
        """

        diff = (target.value - current.value) % 4

        if diff == 0:
            return []

        if diff == 1:
            return [Action.RIGHT]

        if diff == 3:
            return [Action.LEFT]

        return [Action.TURN_AROUND]

    def actions_from_path(
        self,
        path: list[tuple[int, int]],
        initial_heading: Heading,
    ) -> list[Action]:
        """
        Convert a cell path into executable actions.

        Example
        -------
        Path:
            [(0,0), (0,1), (1,1)]

        Output:
            [FORWARD, RIGHT, FORWARD]
        """

        if len(path) < 2:
            return []

        actions: list[Action] = []

        heading = initial_heading

        for current, nxt in zip(path[:-1], path[1:]):

            desired = self.heading_between(
                current,
                nxt,
            )

            actions.extend(
                self.turn_actions(
                    heading,
                    desired,
                )
            )

            actions.append(Action.FORWARD)

            heading = desired

        return actions