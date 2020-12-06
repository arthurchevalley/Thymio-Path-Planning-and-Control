import time

import numpy as np
import os
from src.thymio.Thymio import Thymio
from dotenv import load_dotenv
from threading import Timer

load_dotenv()


def move(thymio: Thymio, l_speed_ratio=0.5, r_speed_ratio=0.5, verbose: bool = False):
    """
    Move the robot's wheels correctly. Manages the negative speed well.
    Once this function is called the robot will continue forever if no further implementation is used.
    Wrap this function to set the conditions for moving.

    :param thymio:      the class to which the robot is referred to
    :param r_speed_ratio:     left speed
    :param l_speed_ratio:     right speed
    :param verbose:     printing the speed in the terminal
    """
    # Changing negative values to the expected ones with the bitwise complement
    l_speed = int(l_speed_ratio * int(os.getenv("LEFT_WHEEL_SCALING")))
    r_speed = int(r_speed_ratio * int(os.getenv("RIGHT_WHEEL_SCALING")))

    # Printing the speeds if requested
    if verbose:
        print("\t\t Setting speed : ", l_speed, r_speed)

    l_speed = l_speed if l_speed >= 0 else 2 ** 16 + l_speed
    r_speed = r_speed if r_speed >= 0 else 2 ** 16 + r_speed
    thymio.set_var("motor.left.target", l_speed)
    thymio.set_var("motor.right.target", r_speed)


def stop(thymio: Thymio, verbose=False):
    """
    Stop the robot.

    :param thymio:      the class to which the robot is referred to
    :param verbose:     printing the stop command in the terminal
    """
    # Printing the speeds if requested
    if verbose:
        print("\t\t Stopping")

    thymio.set_var("motor.left.target", 0)
    thymio.set_var("motor.right.target", 0)


def rotate_time(angle: float, speed_ratio=1):
    """
    Computes the speed ratios and turn time for a certain angle.

    param angle: angle in degrees
    param speed_ratio: ratio to scale the speed

    return: speed norms for left and right wheel, time constant for turning at 100 speed
    """

    left_dir = -speed_ratio * int(np.sign(angle))
    right_dir = speed_ratio * int(np.sign(angle))
    turn_time = float(os.getenv("HALF_TURN_TIME")) * abs(angle) / 180.0 / speed_ratio  # speed of 100
    return left_dir, right_dir, turn_time


def advance_time(distance: float, speed_ratio=1):
    """
    Computes the speed ratios and time for a certain distance.

    param distance: distance in centimeter
    param speed_ratio: ratio to scale the speed

    return: speed norms for left and right wheel, time constant for advancing at 100 speed
    """
    left_dir = speed_ratio * int(np.sign(distance))
    right_dir = speed_ratio * int(np.sign(distance))
    distance_time = float(os.getenv("DISTANCE_TIME")) * abs(distance) / speed_ratio
    return left_dir, right_dir, distance_time


def rotate_thread(thymio: Thymio, angle: float, verbose: bool = False, function=stop, args=None, kwargs=None):
    """
    Rotates of the desired angle by using a timer on a parallel thread.

    :param function:    function to execute at the end of rotation, default stop
    :param args:        array of non-keyworded arguments of function
    :param kwargs:      set of keyworded arguments
    :param thymio:      the class to which the robot is referred to
    :param angle:       angle in radians by which we want to rotate, positive or negative
    :param verbose:     printing the speed in the terminal

    :return: timer to check if it is still alive or not
    """
    args_f = args if args is not None else [thymio]
    kwargs_f = kwargs if kwargs is not None else {}
    l_speed, r_speed, turn_time = rotate_time(angle)

    # Printing the speeds if requested
    if verbose:
        # print("\t\t Rotate speed & time : ", l_speed, r_speed, turn_time)
        print("\t\t Rotate of degrees : ", angle)

    timer = Timer(interval=turn_time, function=function, args=args_f, kwargs=kwargs_f)
    move(thymio, l_speed, r_speed)
    timer.start()
    return timer


def advance_thread(thymio: Thymio, distance: float, speed_ratio: int = 1, verbose: bool = False, function=stop,
                   args=None,
                   kwargs=None):
    """
    Moves straight of a desired distance by using a timer on a parallel thread.

    :param kwargs:      function to execute at the end of advancing, default stop
    :param args:        array of non-keyworded arguments of function
    :param function:    set of keyworded arguments
    :param thymio:      the class to which the robot is referred to
    :param distance:    distance in cm by which we want to move, positive or negative
    :param speed_ratio:       the speed factor at which the robot goes
    :param verbose:     printing the speed in the terminal

    :return: timer to check if it is still alive or not
    """
    args_f = args if args is not None else [thymio]
    kwargs_f = kwargs if kwargs is not None else {}
    l_speed, r_speed, distance_time = advance_time(distance, speed_ratio)

    # Printing the speeds if requested
    if verbose:
        # print("\t\t Advance speed & time : ", l_speed, r_speed, distance_time)
        print("\t\t Advance of cm: ", distance)

    timer = Timer(interval=distance_time, function=function, args=args_f, kwargs=kwargs_f)
    move(thymio, l_speed, r_speed)
    timer.start()
    return timer


def advance(thymio: Thymio, distance: float, speed_ratio: int = 1, verbose: bool = False):
    """
    Moves straight of a desired distance without using a parallel thread

    :param thymio:      the class to which the robot is referred to
    :param distance:    distance in cm by which we want to move, positive or negative
    :param speed_ratio:       the speed factor at which the robot goes
    :param verbose:     printing the speed in the terminal

    :return: timer to check if it is still alive or not
    """
    left_dir, right_dir, distance_time = advance_time(distance, speed_ratio)

    # Printing the speeds if requested
    if verbose:
        # print("\t\t Advance speed & time : ", l_speed, r_speed, distance_time)
        print("\t\t Advance of cm: ", distance)

    move(thymio, left_dir, right_dir)
    time.sleep(distance_time)
    stop(thymio)


def rotate(thymio: Thymio, angle: float, verbose: bool = False):
    """
    Rotates of the desired angle without using a parallel thread

    :param thymio:      the class to which the robot is referred to
    :param angle:       angle in radians by which we want to rotate, positive or negative
    :param verbose:     printing the speed in the terminal

    :return: timer to check if it is still alive or not
    """

    left_dir, right_dir, turn_time = rotate_time(angle)

    # Printing the speeds if requested
    if verbose:
        # print("\t\t Rotate speed & time : ", l_speed, r_speed, turn_time)
        print("\t\t Rotate of degrees : ", angle)

    move(thymio, left_dir, right_dir)
    time.sleep(turn_time)
    stop(thymio)
