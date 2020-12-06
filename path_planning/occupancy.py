import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from src.path_planning.a_star import A_Star
import time
# constants
LENGTH = 29
WIDTH = 32

LOCALIZATION = 0
OCCUPANCY = 1
FREE = 0
OCCUPIED = 1


def display_map(grid, type_map):
    """
    Display a map (either localization grid or occupancy grid)

    :param grid: 2D matrix containing the values of each cell in the map
    :param type_map: specify the type of map  and can take 2 values (LOCALIZATION or OCCUPANCY)

    :return: the fig and ax objects.
    """
    fig, ax = plt.subplots(figsize=(7, 7))

    major_ticks_x = np.arange(0, WIDTH, 5)
    minor_ticks_x = np.arange(0, WIDTH, 1)
    major_ticks_y = np.arange(0, LENGTH, 5)
    minor_ticks_y = np.arange(0, LENGTH, 1)
    ax.set_xticks(major_ticks_x)
    ax.set_xticks(minor_ticks_x, minor=True)
    ax.set_yticks(major_ticks_y)
    ax.set_yticks(minor_ticks_y, minor=True)
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)
    ax.set_ylim([0, (LENGTH - 1)])
    ax.set_xlim([0, (WIDTH - 1)])
    ax.grid(True)

    if type_map == OCCUPANCY:
        # Select the colors with which to display obstacles and free cells
        cmap = colors.ListedColormap(['white', 'red'])

        # Displaying the map
        # ax.imshow(grid, cmap=cmap, extent=[0, 42, 0, 45])
        ax.imshow(grid, cmap=cmap)
        plt.title("Map : free cells in white, occupied cells in red")

    elif type_map == LOCALIZATION:
        cmap = colors.ListedColormap(['white', 'black'])

        # Displaying the map
        ax.imshow(grid, cmap=cmap, extent=[0, (WIDTH - 1), 0, (LENGTH - 1)])
        plt.title("Localization grid")

    return fig, ax


def display_global_path(start, goal, path, occupancy_grid):
    # Displaying the map
    fig_astar, ax_astar = display_map(occupancy_grid, OCCUPANCY)
    # ax_astar.imshow(occupancy_grid.transpose(), cmap=cmap)

    # Plot the best path found and the list of visited nodes
    ax_astar.plot(path[0], path[1], marker="o", color='blue')
    ax_astar.scatter(start[0], start[1], marker="o", color='green', s=200)
    ax_astar.scatter(goal[0], goal[1], marker="o", color='purple', s=200)
    # ax_astar.set_ylim(ax_astar.get_ylim()[::-1])

    ax_astar.set_ylabel('x axis')
    ax_astar.set_xlabel('y axis')
    plt.figure()
    plt.show()


def path_to_command_thymio(path):
    RIGHT = 0
    LEFT = 1
    STRAIGHT = 2

    current_x = path[0][0]
    current_y = path[1][0]

    next_x = path[0][1]
    next_y = path[1][1]

    # next-prev
    delta_x = next_x - current_x
    delta_y = next_y - current_y

    # delat_x = 0 and delta_y = -/+ 1 (or delat_x = -/+ 1 and delta_y = 0): go straight
    turn = STRAIGHT

    # delat_x = -1 and delta_y = 1 (or delat_x = 1 and delta_y = -1): turn to the right
    if delta_x * delta_y < 0:
        turn = RIGHT

    # delat_x = -1 and delta_y = -1 (or delat_x = 1 and delta_y = 1): turn to the left
    if delta_x * delta_y == 1:
        turn = LEFT

    new_path = np.array([path[0][1:], path[1][1:]])

    return turn, new_path


def full_path_to_points(path):
    """
    Concatenates the path if there are multiple points on the same line.

    :param path: numpy array of numpy arrays with points in x and y axis
    :return:    concatenated numpy arrays
    """
    points_x = [path[0][0]]
    points_y = [path[1][0]]

    new_path = path
    prev_turn, new_path = path_to_command_thymio(new_path)

    for i in range(len(new_path[0]) - 1):

        new_turn, new_path = path_to_command_thymio(new_path)

        if new_turn != prev_turn:
            points_x.append(path[0][i + 1])
            points_y.append(path[1][i + 1])

        prev_turn = new_turn

    points_x.append(path[0][-1])
    points_y.append(path[1][-1])
    points = [points_x, points_y]

    return points


def display_occupancy(final_occupancy_grid, position, goal):
    # Run the A* algorithm
    x = round(position[0] / 2.5) - 1
    y = round(position[1] / 2.5) - 1
    new_pos = (x, y)
    print("start: ", new_pos)
    print("goal: ", goal)
    path = A_Star(new_pos, goal, final_occupancy_grid)  # all steps in path
    path = np.array(path).reshape(-1, 2).transpose()
    new_path = full_path_to_points(path)  # concatenated path
    display_global_path(new_pos, goal, new_path, final_occupancy_grid.transpose())
    full_path = np.delete(path, 0, 1)
    # full_path.tolist()
    new_path = np.delete(new_path, 0, 1)
    # new_path.tolist()
    print("path", new_path)
    return new_path, full_path
