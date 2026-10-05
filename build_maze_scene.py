from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


PROJECT_ROOT = Path(__file__).parent

ROBOT_XML = PROJECT_ROOT / "models" / "micromouse.xml"
MAZE_GRID = PROJECT_ROOT / "environments" / "fixed_maze.npy"
OUTPUT_XML = PROJECT_ROOT / "models" / "fixed_maze.xml"


# =========================================================
# PHYSICAL MAZE PARAMETERS
# =========================================================

CELL_SIZE = 0.60

WALL_THICKNESS = 0.09
WALL_HEIGHT = 0.30


# =========================================================
# COLORS
# =========================================================

# Maze walls
WALL_COLOR = "0.05 0.25 0.85 1"

# Entrance / exit
START_COLOR = "0.1 0.85 0.2 1"
EXIT_COLOR = "0.95 0.1 0.1 1"

# Robot chassis
ROBOT_COLOR = "0.9 0.05 0.05 1"

# Robot wheels
WHEEL_COLOR = "0.03 0.03 0.03 1"


# =========================================================
# LOAD BASE ROBOT MODEL
# =========================================================

tree = ET.parse(ROBOT_XML)
root = tree.getroot()

worldbody = root.find("worldbody")

if worldbody is None:
    raise RuntimeError("worldbody not found")


# =========================================================
# WHITE ENVIRONMENT
# =========================================================

asset = root.find("asset")

if asset is None:
    asset = ET.Element("asset")
    root.insert(0, asset)


# Remove old skyboxes.

for texture in list(asset.findall("texture")):

    if texture.get("type") == "skybox":
        asset.remove(texture)


# White skybox.

ET.SubElement(
    asset,
    "texture",
    {
        "name": "white_background",
        "type": "skybox",
        "builtin": "flat",
        "rgb1": "1 1 1",
        "rgb2": "1 1 1",
        "width": "256",
        "height": "256",
    },
)


# =========================================================
# WHITE FLOOR
# =========================================================

floor = worldbody.find(
    "./geom[@name='floor']"
)

if floor is not None:

    floor.set(
        "rgba",
        "0.00 0.00 0.00 1"
    )


# =========================================================
# LOAD FIXED MAZE
# =========================================================

grid = np.load(MAZE_GRID)

grid_rows, grid_cols = grid.shape

rows = (grid_rows - 1) // 2
cols = (grid_cols - 1) // 2

print(f"Maze: {rows} x {cols} cells")


# =========================================================
# COORDINATE SYSTEM
# =========================================================
#
# Cell (0,0) is the TOP-LEFT cell.
#
# Physical maze is centered around (0,0).
#
#             +Y
#              ↑
#
#       (0,0) ────────────
#         │
#         │
#         ↓
#
# X increases to the right.
# Y decreases toward the bottom.
#
# This makes the maze and markers use exactly
# the same coordinate system.
# =========================================================

def cell_center(row, col):

    x = (
        col - (cols - 1) / 2
    ) * CELL_SIZE

    y = (
        (rows - 1) / 2 - row
    ) * CELL_SIZE

    return x, y


# =========================================================
# ROBOT
# =========================================================

robot = worldbody.find(
    "./body[@name='robot']"
)

if robot is None:
    raise RuntimeError("Robot body not found")


start_x, start_y = cell_center(0, 0)


# Robot starts just outside the maze entrance.
# The maze's west boundary is at x = -2.40 m.
# Chassis half-length is 0.10 m.
# Therefore x = -2.50 places its front exactly
# at the entrance.

robot_start_x = start_x - 0.50

robot.set(
    "pos",
    f"{robot_start_x} {start_y} 0.06"
)

robot.set(
    "euler",
    "0 0 180"
)

robot.set(
    "euler",
    "0 0 180"
)


# Make robot visible.

for geom in robot.findall(".//geom"):

    if geom.get("name") == "chassis":

        geom.set(
            "rgba",
            ROBOT_COLOR
        )

    elif "wheel" in geom.get("name", ""):

        geom.set(
            "rgba",
            WHEEL_COLOR
        )


# =========================================================
# WALL CREATION
# =========================================================

def add_wall(
    name,
    x,
    y,
    size_x,
    size_y
):

    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": name,
            "type": "box",
            "pos": (
                f"{x} "
                f"{y} "
                f"{WALL_HEIGHT / 2}"
            ),
            "size": (
                f"{size_x} "
                f"{size_y} "
                f"{WALL_HEIGHT / 2}"
            ),
            "mass": "0",
            "rgba": WALL_COLOR,
            "friction": "1 0.005 0.0001",
        },
    )


# =========================================================
# HORIZONTAL WALLS
# =========================================================
#
# A horizontal boundary exists at:
#
# grid rows:
#
# 0, 2, 4, 6, ... 2*rows
#
# We scan each boundary and MERGE consecutive walls
# into one long physical wall.
# =========================================================

for boundary in range(rows + 1):

    grid_r = 2 * boundary

    c = 0

    while c < cols:

        if grid[grid_r, 2 * c + 1] != 1:

            c += 1
            continue

        start_col = c

        while (
            c < cols
            and grid[grid_r, 2 * c + 1] == 1
        ):

            c += 1

        end_col = c - 1

        # Physical X center of run.

        x_start, _ = cell_center(
            0,
            start_col
        )

        x_end, _ = cell_center(
            0,
            end_col
        )

        x = (
            x_start + x_end
        ) / 2

        # Boundary Y.

        if boundary == 0:

            y = (
                (rows - 1) / 2
                + 0.5
            ) * CELL_SIZE

        else:

            y = (
                (rows - 1) / 2
                - (boundary - 1)
                - 0.5
            ) * CELL_SIZE

        length = (
            (end_col - start_col + 1) * CELL_SIZE
            + WALL_THICKNESS
        )

        add_wall(
            f"hwall_{boundary}_{start_col}",
            x,
            y,
            length / 2,
            WALL_THICKNESS / 2,
        )


# =========================================================
# VERTICAL WALLS
# =========================================================
#
# Vertical boundaries:
#
# grid columns:
#
# 0, 2, 4, ... 2*cols
# =========================================================

for boundary in range(cols + 1):

    grid_c = 2 * boundary

    r = 0

    while r < rows:

        if grid[2 * r + 1, grid_c] != 1:

            r += 1
            continue

        start_row = r

        while (
            r < rows
            and grid[2 * r + 1, grid_c] == 1
        ):

            r += 1

        end_row = r - 1

        _, y_start = cell_center(
            start_row,
            0
        )

        _, y_end = cell_center(
            end_row,
            0
        )

        y = (
            y_start + y_end
        ) / 2

        # Boundary X.

        if boundary == 0:

            x = (
                -(
                    cols - 1
                ) / 2
                - 0.5
            ) * CELL_SIZE

        else:

            x = (
                -(
                    cols - 1
                ) / 2
                + (boundary - 1)
                + 0.5
            ) * CELL_SIZE

        length = (
            (end_row - start_row + 1) * CELL_SIZE
            + WALL_THICKNESS
        )

        add_wall(
            f"vwall_{start_row}_{boundary}",
            x,
            y,
            WALL_THICKNESS / 2,
            length / 2,
        )


# =========================================================
# START MARKER
# =========================================================
#
# Entrance is on WEST side of cell (0,0).
#
# Put the marker AT the entrance opening,
# not in the middle of the maze.
# =========================================================

entrance_x = robot_start_x

ET.SubElement(
    worldbody,
    "geom",
    {
        "name": "start_marker",
        "type": "box",
        "pos": (
            f"{entrance_x} "
            f"{start_y} "
            f"0.004"
        ),
        "size": "0.07 0.18 0.004",
        "rgba": START_COLOR,
        "contype": "0",
        "conaffinity": "0",
    },
)


# =========================================================
# EXIT MARKER
# =========================================================

exit_row = rows - 1
exit_col = cols - 1

exit_x, exit_y = cell_center(
    exit_row,
    exit_col
)

# Exit is on EAST side.

exit_marker_x = (
    exit_x + CELL_SIZE / 2
)

ET.SubElement(
    worldbody,
    "geom",
    {
        "name": "exit_marker",
        "type": "box",
        "pos": (
            f"{exit_marker_x} "
            f"{exit_y} "
            f"0.004"
        ),
        "size": "0.07 0.18 0.004",
        "rgba": EXIT_COLOR,
        "contype": "0",
        "conaffinity": "0",
    },
)


# =========================================================
# SAVE
# =========================================================

ET.indent(
    tree,
    space="    "
)

tree.write(
    OUTPUT_XML,
    encoding="utf-8",
    xml_declaration=True,
)


print()
print("Scene created:")
print(OUTPUT_XML)

print()
print(
    f"Physical maze: "
    f"{cols * CELL_SIZE:.2f} m × "
    f"{rows * CELL_SIZE:.2f} m"
)

print(
    f"Robot start: "
    f"({robot_start_x:.3f}, {start_y:.3f})"
)

print(
    f"Exit cell: "
    f"({exit_x:.3f}, {exit_y:.3f})"
)

print(
    f"Entrance marker: "
    f"({entrance_x:.3f}, {start_y:.3f})"
)

print(
    f"Exit marker: "
    f"({exit_marker_x:.3f}, {exit_y:.3f})"
)