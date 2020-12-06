import time
from enum import Enum
import numpy as np
from src.kalman.kalmann_filter import KalmanHandler
from src.sensors.state import SensorHandler
from src.thymio.Thymio import Thymio
from src.displacement.movement import stop, advance_time, move, rotate_time


class EventEnum(Enum):
    """
    This is a class based on enumeration to define constants in a clean way.
    """
    RIGHT = 0
    LEFT = 1


class ObstacleAvoidance:
    """
    Local avoidance class that manages physical obstacles as soon as it detects them
    """

    def __init__(self, thymio: Thymio, kalman_handler, full_path, final_occupancy_grid, distance_avoidance=2.5,
                 angle_avoidance=5.0, square=2.5, wall_threshold=3600, clear_thresh=2400):
        """
        Constructor to initialize the sensors, camera, kalman and class variables.

        param thymio: class thymio, reference to the robot
        param kalman_handler: class KalmanHandler that wraps the code of the Kalman for the right execution. It includes
                              the use of the sensors and the camera.
        param full_path: list of all intermediary points to reach the goal
        param final_occupancy_grid: map grid with obstacle size increase
        param distance_avoidance: constant steps to move forward
        param angle_avoidance: angle steps to rotate
        param square: unit square size of the grid map
        param wall_threshold: constant min threshold to detect a wall or not
        param clear_thresh: constant max threshold to detect a wall or not
        """
        # params
        self.final_occupancy_grid = final_occupancy_grid
        self.kalman_handler = kalman_handler

        # constants
        self.SQUARE = square
        self.WALL_THRESHOLD = wall_threshold
        self.CLEAR_THRESH = clear_thresh
        self.ONE_STEP = 1
        self.FOUR_STEPS = 4
        self.DISTANCE_AVOIDANCE = distance_avoidance
        self.ANGLE_AVOIDANCE = angle_avoidance
        self.NO_OBSTACLE_SENSOR = 2000
        self.HALF_TURN = 180
        self.CHECK_OBSTACLE = 30
        self.ANGLE_ADJUSTMENT = 20

        # class
        self.thymio = thymio
        self.full_path = full_path

        # methods
        self.sensor_handler = SensorHandler(thymio)

        # variables
        self.kalman_position = self.kalman_handler.get_camera()
        self.kalman_handler.start_timer()
        self.kalman_position = [self.kalman_position[0] / self.SQUARE, self.kalman_position[1] / self.SQUARE,
                                self.kalman_position[2]]

        # initialisation
        self.__update_path()  # update the full_path
        self.__obstacle_avoidance()  # avoid obstacle
        self.__update_path()  # update the full_path

    def __obstacle_avoidance(self):
        """
        Obstacle avoidance handler. It handles all the obstacle avoidance routine.
        """
        # Determine the direction of avoidance and store it in rotated
        sensor_values = self.sensor_handler.sensor_raw()["sensor"]  # load sensors values

        # case 1:  the robot sees a wall in both side. It chooses the opposite direction of the sensor with the largest
        # value which correspond to the direction of the sensor with the largest distance.
        if (sensor_values[1] > self.WALL_THRESHOLD) and (sensor_values[3] > self.WALL_THRESHOLD):
            if sensor_values[3] > sensor_values[1]:
                rotated = EventEnum.LEFT  # go left
            else:
                rotated = EventEnum.RIGHT  # go right

        # case 2:  the robot sees a wall at its right side. It chooses the opposite direction.
        elif (sensor_values[3] > self.WALL_THRESHOLD) or (sensor_values[4] > self.WALL_THRESHOLD):  # right side
            rotated = EventEnum.LEFT

        # case 3:  the robot sees a wall at its right side. It chooses the opposite direction.
        elif (sensor_values[0] > self.WALL_THRESHOLD) or (sensor_values[1] > self.WALL_THRESHOLD):  # left side
            rotated = EventEnum.RIGHT

        # case 4: the robot sees a wall only with its middle sensor. It chooses to go left by default
        elif sensor_values[2] > self.WALL_THRESHOLD:  # center
            rotated = EventEnum.LEFT

        # case 5: None of the conditions were right. It chooses left by default in order to increase robustness
        else:
            rotated = EventEnum.LEFT

        # If the robot go straight toward the corner of an obstacle, it adjust its angle to avoid the obstacle
        sensor_values = self.sensor_handler.sensor_raw()["sensor"]  # load sensors values
        if (rotated == EventEnum.LEFT) and (sensor_values[3] < self.CLEAR_THRESH) and (
                sensor_values[2] > self.WALL_THRESHOLD):
            self.rotate(self.thymio, self.ANGLE_ADJUSTMENT)  # adjust its angle
        elif (rotated == EventEnum.RIGHT) and (sensor_values[1] < self.CLEAR_THRESH) and (
                sensor_values[2] > self.WALL_THRESHOLD):
            self.rotate(self.thymio, -self.ANGLE_ADJUSTMENT)  # adjust its angle

        # The robot spins on itself until it does not see the obstacle with the sensor of the opposite avoidance
        # direction. It ends up being parallel to the obstacle.
        condition = True
        while condition:
            sensor_values = self.sensor_handler.sensor_raw()["sensor"]
            if rotated == EventEnum.LEFT:
                self.rotate(self.thymio, self.ANGLE_AVOIDANCE)
                if sensor_values[3] <= self.CLEAR_THRESH:
                    break
            else:
                self.rotate(self.thymio, -self.ANGLE_AVOIDANCE)
                if sensor_values[1] <= self.CLEAR_THRESH:
                    break
        # readjustments in order to be parallel
        if rotated == EventEnum.LEFT:
            self.rotate(self.thymio, 15)
        else:
            self.rotate(self.thymio, -15)
        stop(self.thymio)  # stop the robot

        # main loop of obstacle avoidance. It runs until it crosses the global path at the other side of the obstacle
        global_path = False
        while not global_path:
            obstacle, global_path = self.__cote_avoid(rotated)
            if obstacle and rotated == EventEnum.LEFT:  # the robot reached a global obstacle in 2D, it spins of 180
                self.rotate(self.thymio, self.HALF_TURN)  # and try to avoid the obstacle from the other side
                rotated = EventEnum.RIGHT  # change direction of avoidance

            elif obstacle and rotated == EventEnum.RIGHT:  # the robot reached a global obstacle in 2D, it spins of 180
                self.rotate(self.thymio, -self.HALF_TURN)  # and try to avoid the obstacle from the other side
                rotated = EventEnum.LEFT  # change direction of avoidance

    def __cote_avoid(self, rotated):
        """
        follow the wall of the obstacle until it does not detect it anymore

        param rotated: direction of avoidance
        """

        condition = True
        obstacle = False  # Reached an obstacle
        global_path = False  # Reached the full_path at the other side of the obstacle

        # main loop. It stops when he reached the full path or an obstacle
        while condition:
            # check if there is an obstacle before it advances
            obstacle, global_path = self.__check_global_obstacles_and_global_path(
                self.ONE_STEP)
            if obstacle:  # if it finds an obstacle, return to change direction of avoidance
                return obstacle, global_path
            self.advance(self.thymio, self.ONE_STEP * self.DISTANCE_AVOIDANCE)  # advance
            if global_path:  # if it finds the full path, return to leave the local avoidance
                return obstacle, global_path

            # check if the wall is still next to itself
            if rotated == EventEnum.LEFT:
                self.rotate(self.thymio, -self.CHECK_OBSTACLE)
            else:
                self.rotate(self.thymio, self.CHECK_OBSTACLE)
            sensor_values = self.sensor_handler.sensor_raw()["sensor"]  # load sensors values

            # Adjust itself to be parallel again and if the robots did not detect a wall, it runs a routine to find it
            # again.

            # case 1: The robot did not detect the wall
            if (rotated == EventEnum.LEFT) and (sensor_values[4] > self.NO_OBSTACLE_SENSOR):  # adjustments
                self.rotate(self.thymio, self.CHECK_OBSTACLE)
            elif (rotated == EventEnum.RIGHT) and (sensor_values[0] > self.NO_OBSTACLE_SENSOR):  # adjustments
                self.rotate(self.thymio, -self.CHECK_OBSTACLE)

            # case 2: The robot detected the wall
            elif (rotated == EventEnum.LEFT) and (sensor_values[4] < self.NO_OBSTACLE_SENSOR):  # readjustments
                self.rotate(self.thymio, self.CHECK_OBSTACLE)

                # check if there is an obstacle before it advances
                obstacle, global_path = self.__check_global_obstacles_and_global_path(self.FOUR_STEPS)
                if obstacle:  # if it finds an obstacle, return to change direction of avoidance
                    return obstacle, global_path
                self.advance(self.thymio, self.FOUR_STEPS * self.DISTANCE_AVOIDANCE)  # advance
                if global_path:  # if it finds the full path, return to leave the local avoidance
                    return obstacle, global_path

                # turn until it sees the obstacle again
                sensor_values = self.sensor_handler.sensor_raw()["sensor"]  # load sensors values
                while sensor_values[4] < self.NO_OBSTACLE_SENSOR:
                    sensor_values = self.sensor_handler.sensor_raw()["sensor"]  # load sensors values
                    self.rotate(self.thymio, -self.ANGLE_AVOIDANCE)
                self.rotate(self.thymio, self.ANGLE_ADJUSTMENT)
                break

            elif (rotated == EventEnum.RIGHT) and (sensor_values[0] < self.NO_OBSTACLE_SENSOR):
                self.rotate(self.thymio, -self.CHECK_OBSTACLE)  # readjustments

                # check if there is an obstacle before it advances
                obstacle, global_path = self.__check_global_obstacles_and_global_path(self.FOUR_STEPS)
                if obstacle:  # if it finds an obstacle, return to change direction of avoidance
                    return obstacle, global_path
                self.advance(self.thymio, self.FOUR_STEPS * self.DISTANCE_AVOIDANCE)
                if global_path:  # if it finds the full path, return to leave the local avoidance
                    return obstacle, global_path

                # turn until it sees the obstacle again
                sensor_values = self.sensor_handler.sensor_raw()["sensor"]  # load sensors values
                while sensor_values[0] < self.NO_OBSTACLE_SENSOR:
                    sensor_values = self.sensor_handler.sensor_raw()["sensor"]  # load sensors values
                    self.rotate(self.thymio, self.ANGLE_AVOIDANCE)
                self.rotate(self.thymio, -self.ANGLE_ADJUSTMENT)  # adjustments
                break
        return obstacle, global_path

    def __check_global_obstacles_and_global_path(self, length_advance):
        """
        Check if there is an obstacle or a map limit or if it crossed the end of the path

        param length_advance: distance of advancement to check
        """

        # initialisation variable
        global_path = False
        obstacle = False
        x = self.kalman_position[0]
        y = self.kalman_position[1]
        theta = self.kalman_position[2] - 90
        x_discrete = round(x)
        y_discrete = round(y)

        # Find the direction to check in the occupancy grid
        if (theta >= -22.5) and (theta < 22.5):
            dir_x = 1
            dir_y = 0
        elif (theta >= 22.5) and (theta < 67.5):
            dir_x = 1
            dir_y = 1
        elif (theta >= 67.5) and (theta < 112.5):
            dir_x = 0
            dir_y = 1
        elif (theta >= 112.5) and (theta < 157.5):
            dir_x = -1
            dir_y = 1
        elif ((theta >= 157.5) and (theta <= 181)) or ((theta >= -181) and (theta < -157.5)):
            dir_x = -1
            dir_y = 0
        elif (theta >= -157.5) and (theta < -112.5):
            dir_x = -1
            dir_y = -1
        elif (theta >= -112.5) and (theta < -67.5):
            dir_x = 0
            dir_y = -1
        elif (theta >= -67.5) and (theta < -22.5):
            dir_x = 1
            dir_y = -1
        else:
            dir_x = 0
            dir_y = 0
            print(" angle is not between -180 and 180 degrees")

        # compute the middle point of the "checking Square"
        x_next_step = int(x_discrete + length_advance * dir_x)
        y_next_step = int(y_discrete + length_advance * dir_y)

        # Verify the map limits and return obstacle=True if it reached it
        if (x_next_step > 29) or (x_next_step < 2) or (y_next_step > 26) or (y_next_step < 2):
            obstacle = True
            return obstacle, global_path

        # Verify the map obstacles in 2D and return obstacle=True if it reached it
        elif self.final_occupancy_grid[x_next_step][y_next_step] == 1:
            obstacle = True
            return obstacle, global_path

        # compute the "checking Square" 3x3
        approx_position_x = [x_next_step - 1, x_next_step, x_next_step + 1]
        approx_position_y = [y_next_step - 1, y_next_step, y_next_step + 1]

        # if it advances more than 3 squares, check also middle distance for global path
        if length_advance > 3:
            x_next_step_2 = int(x_discrete + 2 * dir_x)
            y_next_step_2 = int(y_discrete + 2 * dir_y)
            approx_position_x_2 = [x_next_step_2 - 1, x_next_step_2, x_next_step_2 + 1]
            approx_position_y_2 = [y_next_step_2 - 1, y_next_step_2, y_next_step_2 + 1]

        # check each point of the path until it finds the global path in the advancement distance
        for k in range(len(self.full_path[0])):
            x_path = self.full_path[0][k]
            y_path = self.full_path[1][k]

            for i in range(len(approx_position_x)):
                for j in range(len(approx_position_y)):
                    x_pos = approx_position_x[i]
                    y_pos = approx_position_y[j]
                    if length_advance > 3:  # check also middle distance for global path
                        x_pos_2 = approx_position_x_2[i]
                        y_pos_2 = approx_position_y_2[j]
                        if x_pos_2 == x_path and y_pos_2 == y_path:  # it crossed the global path
                            global_path = True
                            return obstacle, global_path
                    if x_pos == x_path and y_pos == y_path:  # it crossed the global path
                        global_path = True
                        return obstacle, global_path
        return obstacle, global_path

    def __update_path(self):
        """
        update the full path and keep only remaining point to reach the goal
        """

        # initialisation variable
        x = self.kalman_position[0]
        y = self.kalman_position[1]
        x_discrete = round(x)
        y_discrete = round(y)

        # compute the "searching Square" 5x5
        approx_position_x = [x_discrete - 2, x_discrete - 1, x_discrete, x_discrete + 1, x_discrete + 2]
        approx_position_y = [y_discrete - 2, y_discrete - 1, y_discrete, y_discrete + 1, y_discrete + 2]

        # check each point of the path until it finds the full path
        exit_loop = False
        k_pos = []
        for k in range(len(self.full_path[0])):
            x_path = self.full_path[0][k]
            y_path = self.full_path[1][k]

            exit_for = False
            for i in range(len(approx_position_x)):
                for j in range(len(approx_position_y)):
                    x_pos = approx_position_x[i]
                    y_pos = approx_position_y[j]
                    if x_pos == x_path and y_pos == y_path:  # it crossed the full path
                        k_pos.append(k)
                        exit_for = True
                        exit_loop = True
                        break
                if exit_for:
                    break
            # the loop stops only when it finds all the point of the path around the Thymio.
            if (not exit_for) and exit_loop:
                big_k = k_pos[-1] + 1
                # the updated path only keeps the remaining full path to the goal from the instant
                # position of the thymio
                for i in range(big_k):
                    self.full_path = np.delete(self.full_path, 0, 1)
                return

    def advance(self, thymio: Thymio, distance: float, speed_ratio: int = 1, verbose: bool = False):
        """
        Moves straight of a desired distance
        :param thymio:      the class to which the robot is referred to
        :param distance:    distance in cm by which we want to move, positive or negative
        :param speed_ratio:       the speed factor at which the robot goes
        :param verbose:     printing the speed in the terminal
        :return: timer to check if it is still alive or not
        """
        left_dir, right_dir, distance_time = advance_time(distance, speed_ratio)
        # Printing the speeds if requested
        if verbose:
            print("\t\t Advance of cm: ", distance)

        move(thymio, left_dir, right_dir)
        now = time.time()
        while time.time() - now < distance_time:
            self.kalman_position = self.kalman_handler.get_kalman(False)

        self.kalman_position = self.kalman_handler.get_kalman(True)
        self.kalman_position = [self.kalman_position[0] / self.SQUARE, self.kalman_position[1] / self.SQUARE,
                                self.kalman_position[2]]

    def rotate(self, thymio: Thymio, angle: float, verbose: bool = False):
        """
        Rotates of the desired angle

        :param thymio:      the class to which the robot is referred to
        :param angle:       angle in radians by which we want to rotate, positive or negative
        :param verbose:     printing the speed in the terminal
        :return: timer to check if it is still alive or not
        """
        left_dir, right_dir, turn_time = rotate_time(angle)
        # Printing the speeds if requested
        if verbose:
            print("\t\t Rotate of degrees : ", angle)

        ratio = 2
        move(thymio, left_dir / ratio, right_dir / ratio)
        now = time.time()
        while time.time() - now < turn_time * 2:
            self.kalman_position = self.kalman_handler.get_kalman(False)

        self.kalman_position = self.kalman_handler.get_kalman(True)
        self.kalman_position = [self.kalman_position[0] / self.SQUARE, self.kalman_position[1] / self.SQUARE,
                                self.kalman_position[2]]
