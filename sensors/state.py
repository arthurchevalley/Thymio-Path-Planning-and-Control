import numpy as np
import math
from scipy.interpolate import interp1d
from src.thymio.Thymio import Thymio

# Sensor measurements
sensor_distances = np.array([i for i in range(0, 21)])
sensor_measurements = np.array([5120, 4996, 4964, 4935, 4554, 4018, 3624, 3292, 2987,
                                2800, 2580, 2307, 2039, 1575, 1127, 833, 512, 358, 157, 52, 0])
# sensor_distances = np.array([0, 30])
# sensor_measurements = np.array([5120, 0])

# Thymio outline
center_offset = np.array([5.5, 5.5])
thymio_coords = np.array([[0, 0], [11, 0], [11, 8.5], [10.2, 9.3],
                          [8, 10.4], [5.5, 11], [3.1, 10.5],
                          [0.9, 9.4], [0, 8.5], [0, 0]]) - center_offset

# Sensor positions and orientations
sensor_pos_from_center = np.array(
    [[0.9, 9.4], [3.1, 10.5], [5.5, 11.0], [8.0, 10.4], [10.2, 9.3], [8.5, 0], [2.5, 0]]) - center_offset
sensor_angles = np.array([120, 105, 90, 75, 60, -90, -90]) * math.pi / 180


def sensor_val_to_cm_dist(val: int) -> int:
    """
    Interpolation from sensor values to distances in cm

    :param val: the sensor value that you want to convert to a distance
    :return:    corresponding distance in cm
    """
    if val == 0:
        return np.inf

    f = interp1d(sensor_measurements, sensor_distances)
    return np.asscalar(f(val))


def obstacles_pos_from_sensor_val(sensor_val):
    """
    Returns a list containing the position of the obstacles
    w.r.t the center of the Thymio robot.

    :param sensor_val:     sensor values provided clockwise starting from the top left sensor.
    :return: numpy.array()  that contains the position of the different obstacles
    """
    dist_to_sensor = [sensor_val_to_cm_dist(x) for x in sensor_val]
    dx_from_sensor = [d * math.cos(alpha) for (d, alpha) in zip(dist_to_sensor, sensor_angles)]
    dy_from_sensor = [d * math.sin(alpha) for (d, alpha) in zip(dist_to_sensor, sensor_angles)]
    obstacles_pos = [[x[0] + dx, x[1] + dy] for (x, dx, dy) in
                     zip(sensor_pos_from_center, dx_from_sensor, dy_from_sensor)]
    return np.array(obstacles_pos)


class SensorHandler:
    """
    Get the data from every sensors

    :param thymio:      class of the robot to refer to
    """

    def __init__(self, thymio: Thymio):
        self.thymio = thymio

    def ground(self):
        return {"ground": self.thymio["prox.ground.reflected"]}

    def speed(self):
        return {"left_speed": self.thymio["motor.left.speed"],
                "right_speed": self.thymio["motor.right.speed"]}

    def sensor_raw(self):
        return {"sensor": self.thymio["prox.horizontal"]}

    def all_raw(self):
        """
        Fetch the all the data data of thymio

        :return: return the data of thymio of the time interval
        """
        return {"sensor": self.thymio["prox.horizontal"],
                "ground": self.thymio["prox.ground.reflected"],
                "left_speed": self.thymio["motor.left.speed"],
                "right_speed": self.thymio["motor.right.speed"]}

    def sensor_cm(self):
        val = {"sensor": self.thymio["prox.horizontal"]}
        return obstacles_pos_from_sensor_val(val['sensor'])

    def all_cm(self):
        """
        Fetch the all the data data of thymio and converts sensor to cm

        :return: return the data of thymio of the time interval
        """
        return {"sensor": obstacles_pos_from_sensor_val(self.thymio["prox.horizontal"]),
                "ground": self.thymio["prox.ground.reflected"],
                "left_speed": self.thymio["motor.left.speed"],
                "right_speed": self.thymio["motor.right.speed"]}
