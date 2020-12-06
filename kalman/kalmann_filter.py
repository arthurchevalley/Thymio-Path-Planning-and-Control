import cmath
import os
import threading
import time

import numpy as np
import math

import matplotlib.pyplot as plt
from src.displacement.movement import stop
from src.sensors.state import SensorHandler
from src.thymio.Thymio import Thymio


class Kalman:
    """
    Kalman class that calculates the estimation of the position based on odometry and/or measurements.
    """

    def __init__(self, qx=2.8948e-04, qy=8.2668e-04, qt=2.9e-03, k_delta_sr=2.50e-02, k_delta_sl=1.5e-02):
        """
        Constructor that initializes the class variables.
        recommended values: qx=2.8948e-04, qy=8.2668e-04, qt=2.9e-03, k_delta_sr=1.3400e-02, k_delta_sl=8.3466e-03

        param qx: variance on the x axis
        param qy: variance on the y axis
        param qt: variance on the angle
        param k_delta_sr: variance on the right speed
        param k_delta_sl: variance on the left speed
        """
        self.Ts = 0.1
        self.state_est = [np.array([[0], [0], [0]])]
        self.cov_est = [0.01 * np.ones([3, 3])]  # default, do not tune here
        self.b = 0.095
        self.H = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.stripe_width = 50
        self.qx = qx
        self.qy = qy
        self.qt = qt
        self.k_delta_sr = k_delta_sr
        self.k_delta_sl = k_delta_sl
        self.Q = np.array([[self.qx, 0, 0], [0, self.qy, 0], [0, 0, self.qt]])
        self.R = np.array([[self.k_delta_sr, 0], [0, self.k_delta_sl]])
        plt.ion()
        self.fig, self.ax = plt.subplots()
        plt.show()
        self.cov_all = []
        self.pos_all = []

    def tune_values(self, qx, qy, qt, k_delta_sr, k_delta_sl):
        """
        Allows to change the variance matrices during the execution

        param qx: variance on the x axis
        param qy: variance on the y axis
        param qt: variance on the angle
        param k_delta_sr: variance on the right speed
        param k_delta_sl: variance on the left speed
        """
        self.qx = qx
        self.qy = qy
        self.qt = qt
        self.k_delta_sr = k_delta_sr
        self.k_delta_sl = k_delta_sl
        self.Q = np.array([[self.qx, 0, 0], [0, self.qy, 0], [0, 0, self.qt]])
        self.R = np.array([[self.k_delta_sr, 0], [0, self.k_delta_sl]])

    def __jacobianf_x(self, theta, delta_s, delta_theta):
        """
        Compute the partial derivative of the motion model with respect to the state vector x, evaluated at the current state x and input u 

        :param theta: current orientation of the robot 
        :param delta_s: mean of the travelled distance of the right wheel and the left wheel 
        :param delta_theta: angle increment based on the travelled distance of the right wheel and the left wheel, and the distance between the wheels

        :return: a matrix (np.array) containing the partial derivative evaluated at the current state and input u 
        """
        return np.array(
            [[1, 0, -delta_s * np.sin(theta + delta_theta / 2)], [0, 1, delta_s * np.cos(theta + delta_theta / 2)],
             [0, 0, 1]])

    def __jacobianf_u(self, theta, delta_s, delta_theta):
        """
        Compute the partial derivative of the motion model with respect to the input vector u, evaluated at the current state x and input u 

        :param theta: current orientation of the robot 
        :param delta_s: mean of the travelled distance of the right wheel and the left wheel 
        :param delta_theta: angle increment based on the travelled distance of the right wheel and the left wheel, and the distance between the wheels

        :return: a matrix (np.array) containing the partial derivative evaluated at the current state x and input u 
        """
        return np.array(
            [[1 / 2 * np.cos(theta + delta_theta / 2) - delta_s / (2 * self.b) * np.sin(theta + delta_theta / 2),
              1 / 2 * np.cos(theta + delta_theta / 2) + delta_s / (2 * self.b) * np.sin(theta + delta_theta / 2)],
             [1 / 2 * np.sin(theta + delta_theta / 2) + delta_s / (2 * self.b) * np.cos(theta + delta_theta / 2),
              1 / 2 * np.sin(theta + delta_theta / 2) - delta_s / (2 * self.b) * np.cos(theta + delta_theta / 2)],
             [1 / self.b, -1 / self.b]])

    def plot(self):
        state_pred = self.pos_all
        cov_pred = self.cov_all

        plt.ion()
        fig, ax = plt.subplots()

        px, py = self.plot_covariance_ellipse(state_pred[0], cov_pred[0] / 1000)
        line_v = ax.axvline(x=state_pred[0][0], color="k")
        line_h = ax.axhline(y=state_pred[0][1], color="k")
        ellips, = ax.plot(px, py, "--r", label="covariance matrix")

        max_l = len(state_pred)
        l = [i for i in range(max_l)]
        l.insert(0, -1)
        for i in l:
            px, py = self.plot_covariance_ellipse(state_pred[i], cov_pred[i] / 1000)

            line_v.set_xdata(state_pred[i][0])
            line_h.set_ydata(state_pred[i][1])

            ellips.set_xdata(px)
            ellips.set_ydata(py)
            ax.relim()
            ax.autoscale_view()

            fig.canvas.draw()

            fig.canvas.flush_events()
            # plt.axis([0, 0.725, 0, 0.8])
            plt.show()

            time.sleep(4)

    def plot_covariance_ellipse(self, state_est, cov_est):

        Pxy = cov_est[0:2, 0:2]
        eigval, eigvec = np.linalg.eig(Pxy)

        if eigval[0] >= eigval[1]:
            bigind = 0
            smallind = 1
        else:
            bigind = 1
            smallind = 0

        t = np.arange(0, 2 * math.pi + 0.1, 0.1)
        a = math.sqrt(eigval[bigind])
        b = math.sqrt(eigval[smallind])
        x = [a * math.cos(it) for it in t]
        y = [b * math.sin(it) for it in t]

        angle = math.atan2(eigvec[bigind, 1], eigvec[bigind, 0])
        R = np.array([[math.cos(angle), math.sin(angle)],
                      [-math.sin(angle), math.cos(angle)]])
        fx = R.dot(np.array([[x, y]]))
        px = np.array(fx[0, :] + state_est[0, 0]).flatten()
        py = np.array(fx[1, :] + state_est[1, 0]).flatten()

        return px, py

    def kalman_filter(self, z, state_est_prev, cov_est_prev, delta_sr, delta_sl, measurement):
        """
        Estimates the current state using input sensor data and the previous state
        Everything is in meter and seconds here!

        param z: array representing the measurement (x,y,theta) (coming from the vision sensor)
        param delta_sr: travelled distance for the right wheel (in meters)
        param delta_sl: travelled distance for the left wheel (in meters)
        param state_est_prev: previous state a posteriori estimation
        param cov_est_prev: previous state a posteriori covariance

        return state_est: new a posteriori state estimation
        return cov_est: new a posteriori state covariance
        """
        if z[0] < 0 or z[1] < 0:  # if no camera, just odometry
            measurement = False

        theta = state_est_prev[2]
        delta_s = (delta_sr + delta_sl) / 2
        delta_theta = (delta_sr - delta_sl) / self.b

        Fx = self.__jacobianf_x(theta, delta_s, delta_theta)
        Fu = self.__jacobianf_u(theta, delta_s, delta_theta)

        # Prediction step
        # estimated mean of the state
        z = np.array([[z[0]], [z[1]], [z[2]]])
        state_est_prev = np.array([[state_est_prev[0]], [state_est_prev[1]], [state_est_prev[2]]])

        state_est_a_priori = state_est_prev + np.array(
            [[delta_s * np.cos(theta + delta_theta / 2)], [delta_s * np.sin(theta + delta_theta / 2)], [delta_theta]])

        # Estimated covariance of the state
        cov_est_a_priori = np.dot(Fx, np.dot(cov_est_prev, Fx.T)) + np.dot(Fu, np.dot(self.R, Fu.T))
        """
        print("z: ", z)
        print("state_est_prev: ", state_est_prev)
        print("cov_est_prev: ", cov_est_prev)
        print("delta_sr: ", delta_sr)
        print("delta_sl: ", delta_sl)
        print("Fx", Fx)
        print("Fu", Fu)
        print("state_est_a_priori", state_est_a_priori)
        """

        if measurement:  # odometry et measurements
            # Update step
            # innovation / measurement residual
            i = z - state_est_a_priori

            # Kalman gain (tells how much the predictions should be corrected based on the measurements)
            K = np.dot(cov_est_a_priori, np.linalg.inv(cov_est_a_priori + self.Q))

            # a posteriori estimate
            state_est = state_est_a_priori + np.dot(K, i)
            cov_est = cov_est_a_priori - np.dot(K, cov_est_a_priori)
            """
            print("i", i)
            print(K)
            print("state_est", state_est)
            print("cov_est", cov_est)
            """

        else:  # odometry
            state_est = state_est_a_priori
            cov_est = cov_est_a_priori
        return state_est.flatten().tolist(), cov_est


class KalmanHandler:
    """
    Kalman Handler class that wraps the code of the Kalman for the right execution. It includes the use of the sensors and the camera.
    """

    def __init__(self, thymio: Thymio, camera, interval_sleep=0.05):
        """
        Constructor that initializes the camera, kalman, the sensors and class variables.

        param thymio: class of the robot
        param interval_sleep: time constant to sleep before function loop calls

        Example:
            if recording:
                kalman_handler = KalmanHandler(...)
                kalman_handler.start_recording()
                ...
                kalman_handler.stop_recording()
        """
        self.interval_sleep = interval_sleep
        self.thymio = thymio
        self.kalman = Kalman()
        self.record_left = []
        self.record_right = []
        self.sensor_handler = SensorHandler(self.thymio)
        self.recording = False
        self.kalman_time = 0
        self.thymio_speed_to_mm_s = float(os.getenv("SPEED_80_TO_MM_S"))
        self.camera = camera
        self.covariance = 1.0e-04 * np.ones([3, 3])
        self.kalman_position = [0, 0, 0]
        self.camera_position = [-1, -1, 0]

    def __speeds(self):
        """
        Returns the speed of the robot for left and right wheel.

        :return: left and right speed
        """
        speed = self.sensor_handler.speed()
        r_speed = speed['right_speed']
        l_speed = speed['left_speed']
        l_speed = l_speed if l_speed <= 2 ** 15 else l_speed - 2 ** 16
        r_speed = r_speed if r_speed <= 2 ** 15 else r_speed - 2 ** 16
        return l_speed, r_speed

    def __record_handler(self):
        """
        Manages the speed recording of the robot to get the average speed in kalman.
        """
        l_speed, r_speed = self.__speeds()
        self.record_left.append(l_speed)
        self.record_right.append(r_speed)

        if self.recording:
            time.sleep(self.interval_sleep / 5)  # 5 recordings per kalman call
            self.__record_handler()
        else:
            self.__record_reset()

    def __record_reset(self):
        """
        Empty the arrays of the speeds it stops recording
        """
        self.record_right = []
        self.record_left = []

    def __record_filter(self, threshold):
        """
        Filter the speeds of the robot by removing the elements below the threshold
        """
        self.record_right = filter(lambda number: number < threshold, self.record_right)
        self.record_left = filter(lambda number: number < threshold, self.record_left)

    def start_recording(self):
        """
        Starts a parallel thread to record the speeds in the background.
        """
        self.kalman_time = time.time()
        self.recording = True
        print("START RECORDING")
        threading.Thread(target=self.__record_handler).start()
        time.sleep(self.interval_sleep)

    def stop_recording(self):
        """
        Stops recording the speeds in the background.
        """
        print("STOP RECORDING")
        self.recording = False

    def start_timer(self):
        """
        Start the timer for kalman
        """
        self.kalman_time = time.time()

    def get_kalman(self, measurement: bool):
        """
        Manages the kalman estimation based on the time and speed.

        param measurement: boolean to access the measurements or not in the kalman.

        return: [x, y, theta] converted back in cm and degrees
        """

        # if recording then does the average speed
        if self.recording:
            # if the speeds recorded are empty, stop the kalman
            if not len(self.record_left) and not len(self.record_right):
                return self.kalman_position

            speed_left = sum(self.record_left) / len(self.record_left)
            speed_right = sum(self.record_right) / len(self.record_right)
        # if not recording, takes the actual speed
        else:
            speed_left, speed_right = self.__speeds()

        ts = time.time() - self.kalman_time  # [s]
        delta_sl = speed_left * ts * self.thymio_speed_to_mm_s / 1000  # [m]
        delta_sr = speed_right * ts * self.thymio_speed_to_mm_s / 1000  # [m]

        if measurement:
            # stop(self.thymio)
            self.__camera_handler()

        self.sensor_handler = SensorHandler(self.thymio)  # new class declaration to avoid calling too many times
        self.__record_reset()
        self.start_timer()

        # cm & degrees -> m & rad
        conv_pos = [self.kalman_position[0] / 100, self.kalman_position[1] / 100, np.deg2rad(self.kalman_position[2])]
        conv_cam = [self.camera_position[0] / 100, self.camera_position[1] / 100, np.deg2rad(self.camera_position[2])]
        temp, self.covariance = self.kalman.kalman_filter(conv_cam, conv_pos, self.covariance, delta_sr,
                                                          delta_sl, measurement)
        # m & rad -> cm & degrees
        self.kalman.cov_all.append(temp)
        self.kalman.pos_all.append(self.covariance)
        self.kalman_position = [temp[0] * 100, temp[1] * 100, (np.rad2deg(temp[2]) + 180.0) % 360.0 - 180.0]
        return self.kalman_position

    def __camera_handler(self):
        """
        Camera handler function
        """
        self.camera_position = self.camera.record_project()
        print("camera position", self.camera_position)

    def get_camera(self):
        """
        Sets the robot position with the camera

        return: [x, y, theta] of the camera in degrees & m
        """
        self.__camera_handler()
        self.kalman_position = self.camera_position
        return self.camera_position
