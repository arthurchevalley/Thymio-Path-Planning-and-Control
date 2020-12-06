import numpy as np


def update_path(path, x, y, theta, case_size):
    """
    Do a sequence of  displacement corresponding to the entire global path planning.

    param x: x position robot in cm
    param y: y position robot in cm
    param theta: angle robot in degrees

    return: distance and rotation, needed to go to next step, in cm and degrees
    """
    if len(path[0]) and len(path[1]):
        target_x = path[0][0] * case_size
        target_y = path[1][0] * case_size
        delta_x_cm = target_x - x
        delta_y_cm = target_y - y

        # Relative displacement to target
        delta_r = np.sqrt(delta_x_cm ** 2 + delta_y_cm ** 2)

        # Relative rotation to target
        target_theta_rad = np.arctan2(delta_y_cm, delta_x_cm)
        target_theta_deg = np.rad2deg(target_theta_rad) + 90    # match the grid axis to the robot axis
        delta_theta = target_theta_deg - theta
        delta_theta = (delta_theta + 180.0) % 360.0 - 180.0
        return delta_r, delta_theta
