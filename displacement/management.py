import threading
import time

import numpy as np

from src.displacement.movement import stop, rotate_time, move, advance_time
from src.displacement.planning import update_path
from src.kalman.kalmann_filter import KalmanHandler
from src.local_avoidance.obstacle import ObstacleAvoidance
from src.path_planning.localization import Localization
from src.path_planning.occupancy import display_occupancy, full_path_to_points
from src.thymio.Thymio import Thymio
from src.vision.camera import Camera

#  State of the thymio
FORWARD = 0
TURN = 1
STOP = 2


class EventHandler:
    """
    This class manages all the different scenarios of the robot until it reaches the goal.
    """

    def __init__(self, thymio: Thymio, interval_camera=1, interval_odometry=0.1, interval_sleep=0.08,
                 obstacle_threshold=4100, epsilon_theta=20, epsilon_r=2):
        """
        Constructor of the class EventHandler.

        param thymio: class thymio, reference to the robot
        param interval_camera: time constant necessary to access to kalman odometry and measurement
        param interval_odometry: time constant necessary to access to kalman odometry
        param interval_sleep: time sleep constant before function loop calls
        param obstacle_threshold: condition to go into local avoidance
        param epsilion_theta: the tolerated angle deviation
        param epsilon_r: the tolerated distance deviation


        :return:
        """
        self.thymio: Thymio = thymio
        self.interval_camera = interval_camera
        self.interval_odometry = interval_odometry
        self.interval_sleep = interval_sleep
        self.obstacle_threshold = obstacle_threshold
        self.case_size_cm = 2.5  # [cm]
        self.camera = Camera()
        self.camera.open_camera()
        self.final_occupancy_grid, self.goal = Localization(self.camera).localize()
        self.kalman_handler = KalmanHandler(self.thymio, self.camera)
        self.kalman_position = self.kalman_handler.get_camera()
        self.epsilon_theta = epsilon_theta  # [degrees]
        self.epsilon_r = epsilon_r  # [cm]
        self.path, self.full_path = display_occupancy(self.final_occupancy_grid,
                                                      (self.kalman_position[0], self.kalman_position[1]),
                                                      self.goal)
        # self.kalman_handler.start_recording()
        self.kalman_handler.start_timer()
        self.camera_timer = time.time()
        self.odometry_timer = time.time()
        # self.state = STOP
        self.__global_handler()

    def __global_handler(self):
        """
        Function called in loop until the goal is reached. Kalman, global displacement, local avoidance happens here.
        """
        """
        """
        # odometry and measurement kalman
        if time.time() - self.camera_timer >= self.interval_camera:
            # self.kalman_position = self.kalman_handler.get_camera()
            print("before kalman position", self.kalman_position)
            self.kalman_position = self.kalman_handler.get_kalman(True)
            print("after kalman position", self.kalman_position)
            # self.kalman_handler.stop_recording()
            # self.kalman_handler = KalmanHandler(self.thymio, self.camera)
            # self.kalman_handler.start_recording()
            self.camera_timer = time.time()
            self.odometry_timer = time.time()

        # odometry kalman
        if time.time() - self.odometry_timer >= self.interval_odometry:
            # self.kalman_position = self.kalman_handler.get_camera()
            self.kalman_position = self.kalman_handler.get_kalman(False)
            self.odometry_timer = time.time()

        # get orientation and displacement needed to reach next point of the path
        delta_r, delta_theta = update_path(self.path, self.kalman_position[0], self.kalman_position[1],
                                           self.kalman_position[2],
                                           self.case_size_cm)
        # print("delta_r, delta_theta", delta_r, delta_theta)
        # TODO add scaling to slow down when close to goal

        # Apply rotation
        if abs(delta_theta) > self.epsilon_theta:
            if abs(delta_r) < self.epsilon_r:
                print("Arrived to goal (from rotating)")
                stop(self.thymio)
                self.path = np.delete(self.path, 0, 1)  # removes the step done from the non-concatenated lists
            left_dir, right_dir, turn_time = rotate_time(delta_theta)
            left_dir = left_dir * 0.5
            right_dir = right_dir * 0.5
            if abs(delta_theta) < 20:  # turn less quickly near epsilon_theta
                left_dir = left_dir * 0.5
                right_dir = right_dir * 0.5
            move(self.thymio, left_dir, right_dir)

        # Apply displacement
        elif abs(delta_r) > self.epsilon_r:
            # print("done rotating")
            left_dir, right_dir, distance_time = advance_time(delta_r)
            left_dir = left_dir * 0.5
            right_dir = right_dir * 0.5
            move(self.thymio, left_dir, right_dir)

            # check if local avoidance needed
            sensor_values = self.kalman_handler.sensor_handler.sensor_raw()
            if np.amax(sensor_values["sensor"][0:4]).astype(int) >= self.obstacle_threshold:
                stop(self.thymio)
                # self.kalman_handler.stop_recording()
                self.__local_handler()
                # self.kalman_handler.start_recording()
                self.camera_timer = time.time()
                self.odometry_timer = time.time()
        else:
            # point in the path has been reached
            # self.state = STOP
            stop(self.thymio)
            print("REMOVE POINTS", self.path[0][0], self.path[1][0])
            self.path = np.delete(self.path, 0, 1)  # removes the step done from the non-concatenated lists

        # if there still exist a path, iterates once more
        if len(self.path[0]):
            time.sleep(self.interval_sleep)
            self.__global_handler()

        # no more path, go back to main
        else:
            # self.kalman_handler.stop_recording()
            self.camera.close_camera()
            stop(self.thymio)
            # self.kalman_handler.kalman.plot()
            with open('cov_all.txt', 'w') as f:
                for item in self.kalman_handler.kalman.cov_all:
                    f.write("%s," % item)
            f.close()
            with open('pos_all.txt', 'w') as f:
                for item in self.kalman_handler.kalman.pos_all:
                    f.write("%s," % item)
            f.close()

    def __local_handler(self):
        """
        Local avoidance handler that updates the path after done avoiding.
        """
        obstacle = ObstacleAvoidance(self.thymio, self.kalman_handler, self.full_path, self.final_occupancy_grid)
        self.full_path = obstacle.full_path
        self.kalman_position = obstacle.kalman_position
        if len(self.full_path[0]) < 2:
            self.full_path = np.array([[self.goal[0]], [self.goal[1]]])
            return
        self.path = full_path_to_points(self.full_path)  # concatenated path
