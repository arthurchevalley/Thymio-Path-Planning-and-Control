import os

from src.path_planning.occupancy import display_map
from src.thymio.Thymio import Thymio
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.vision.camera import Colors, Camera

LENGTH = 32
WIDTH = 29

def resize(final_grid, alpha, beta):
    """
    Resize to the desired grid size the camera picture
    :param final_grid:      The picture to resize
    :param alpha:           Sharpen's contrast control to resize the grid
    :param beta:            Sharpen's brightness control to resize the grid
    """

    # init the filter before actual resize
    adjusted = cv2.convertScaleAbs(final_grid, alpha, beta)
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpen = cv2.filter2D(final_grid, -1, sharpen_kernel)

    # resize the map to the wanted size
    map_w_border_row = WIDTH + 2
    map_w_border_col = LENGTH + 2
    vis_map = cv2.resize(sharpen, (map_w_border_col, map_w_border_row), cv2.INTER_AREA)
    return vis_map


class Localization:

    def __init__(self, camera):
        # init value for value setting
        self.color_threshold = 150
        self.zero_init = 0
        self.b, self.g, self.r = (0, 1, 2)
        self.Border = 0
        self.goal, self.thymio = (0, 1)
        self.x, self.y = (0, 1)
        self.colors = Colors()
        self.camera = camera

        # constants
        self.LOCALIZATION = 0
        self.OCCUPANCY = 1
        self.FREE = 0
        self.OCCUPIED = 1

        #####################################
        # dans prog tif passer de 45 a 44 et 42 a 41?
        #####################################
        # shapren parameter
        self.alpha = 1.5  # Contrast control (1.0-3.0)
        self.beta = 0  # Brightness control (0-100)

    def rotate(self, vis_map):
        """
        Rotate the map in order to put it in the right orientation
        :param vis_map:     picture to rotate
        """

        # computing of the green mask to find the correct orientation of the map
        hsv = cv2.cvtColor(vis_map, cv2.COLOR_BGR2HSV)
        mask_green = cv2.inRange(hsv, self.colors.low_green, self.colors.up_green)
        grid_corner = np.array(mask_green)
        n_rows, n_cols = grid_corner.shape

        # turning the whole map until the orientation is good or let it in the initial position
        test_full_rot = 0
        rotation_marker_row = 0
        rotation_marker_col = n_cols - 1

        while grid_corner[rotation_marker_row][n_cols - 1] == 0:
            grid_corner = np.rot90(grid_corner)
            vis_map = np.rot90(vis_map)
            n_rows, n_cols = grid_corner.shape
            test_full_rot = test_full_rot + 1
            if test_full_rot == 4:
                break

        # final world map with goal, thymio and obstacle
        world = vis_map[1:n_rows - 1, 1:n_cols - 1]
        return world

    def detect_object(self, world):
        """
        Fonction to detect the global obstacles and the goal
        :param world:       global map of the world from the camera after resize and rotation
        """

        # create the mask to see the obstacle
        world_hsv = cv2.cvtColor(world, cv2.COLOR_BGR2HSV)
        mask_red = cv2.inRange(world_hsv, self.colors.low_red, self.colors.up_red)
        occupancy_grid = np.array(mask_red)
        plt.figure()
        plt.imshow(occupancy_grid)
        plt.show()
        world_rows, world_cols, _ = world.shape

        #  create the mask in order to find the goal
        world_hsv = cv2.cvtColor(world, cv2.COLOR_BGR2HSV)
        mask_goal = cv2.inRange(world_hsv, self.colors.low_blue, self.colors.up_blue)
        plt.figure()
        plt.imshow(mask_goal)
        plt.show()
        goal_x, goal_y = (15, 15)  # goal by default

        # look for the obstacle and increase there size
        for i in range(world_rows):
            for j in range(world_cols):
                occupancy_grid[i][j] = int(occupancy_grid[i][j] / 255)
                if mask_goal[i][j] > 200:
                    goal_x, goal_y = (i, j)
        object_grid = [[goal_x, goal_y]]
        return object_grid, occupancy_grid

    def vision(self, image):
        """
        Main function handling the pictures capture and preparation before analysis
        :param image:       Picture took with the webcam
        """

        final_grid = Camera().detect_and_rotate(image)
        vis_map = resize(final_grid, self.alpha, self.beta)
        world = self.rotate(vis_map)
        plt.figure()
        plt.imshow(world)
        plt.show()
        object_grid, occupancy_grid = self.detect_object(world)

        return object_grid, occupancy_grid, world

    def display_global_path(self, start, goal, path, occupancy_grid):
        """
        Function to plot the global path
        :param start:           Thymio position at the beginning
        :param goal:            Goal position
        :param path:            Optimal path computed with the A* algorithm
        :param occupancy_grid:  Grid with all the global obstacles
        """
        # Displaying the map
        fig_astar, ax_astar = display_map(occupancy_grid, self.OCCUPANCY)
        # ax_astar.imshow(occupancy_grid.transpose(), cmap=cmap)

        # Plot the best path found and the list of visited nodes
        ax_astar.plot(path[0], path[1], marker="o", color='orange')
        ax_astar.scatter(start[0], start[1], marker="o", color='green', s=200)
        ax_astar.scatter(goal[0], goal[1], marker="o", color='purple', s=200)
        ax_astar.set_ylim(ax_astar.get_ylim()[::-1])

    def increased_obstacles_map(self, occupancy_grid):
        """
        Increase the obstalce size in order to compute A* algorithm without being concern by the thymio size.
                                our case, we increase by 3 square.
        :param occupancy_grid:  Grid with all the global obstacles
        """
        nb_rows, nb_cols = occupancy_grid.shape
        #  increase the total map size by 3
        increased_occupancy_grid = np.zeros([nb_rows + 6, nb_cols + 6])
        for i in range(nb_rows):
            for j in range(nb_cols):

                if occupancy_grid[i, j] == self.OCCUPIED:
                    increased_occupancy_grid[i:i + 7, j:j + 7] = np.ones([7, 7])

        #  Return the reduce to intial size but with increase obstacles map
        final_occupancy_grid = increased_occupancy_grid[3:LENGTH + 3, 3:WIDTH + 3]
        return final_occupancy_grid

    def localize(self):
        """
        Main function handling the image analysis and thymio localisation with the camer
        """
        # open the video and saves the first image
        _, image = self.camera.cap.read()

        """
        cv2.imwrite('C:/Users/Olivier/Documents/EPFL 2020-2021/Basics of mobile robotics/Project/images/init.jpg',
                    frame)
        image = cv2.imread(
            'C:/Users/Olivier/Documents/EPFL 2020-2021/Basics of mobile robotics/Project/images/init.jpg')
        """

        #  Call the vision function in order to have the grid with the obstacle and the goal coordinate
        object_grid, occupancy_grid, world = self.vision(image)

        # change to the right coordinate format
        occupancy_grid = (np.flipud(occupancy_grid)).transpose()
        final_occupancy_grid = self.increased_obstacles_map(occupancy_grid)

        display_map(final_occupancy_grid.transpose(), 1)

        #  goal coordinate
        goal_x = object_grid[self.goal][self.y]
        goal_y = WIDTH - object_grid[self.goal][self.x] - 1
        goal = (goal_x, goal_y)
        print(goal)
        # Run the A* algorithm
        # path = A_Star(start, goal, final_occupancy_grid)
        # path = np.array(path).reshape(-1, 2).transpose()
        return final_occupancy_grid, goal
