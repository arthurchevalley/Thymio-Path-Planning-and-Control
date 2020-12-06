import os
import sys
import pprint
import time
import numpy as np
from src.displacement.management import EventHandler
from src.displacement.movement import move, stop
from src.sensors.state import SensorHandler
from src.sensors.tuning import MotionTuning
from src.thymio.Thymio import Thymio
from dotenv import load_dotenv
from src.vision.camera import Camera
from src.local_avoidance.obstacle import ObstacleAvoidance

# Adding the src folder in the current directory as it contains the script with the Thymio class
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
load_dotenv()


def print_thymio(thymio: Thymio):
    """
    Print the variables of Thymio
    :param thymio: The file location of the spreadsheet
    """
    print('All Thymio instance attributes:')
    pprint.pprint(dir(thymio))
    variables = thymio.variable_description()  # see what the different read-write variables that you can access are
    print('\nVariables of Thymio:')
    for var in variables:
        print(var)


def main():
    """
    Main function that that is used to run the code
    :return:
    """

    """
    cam = Camera()
    cam.open_camera()
    cam.test_camera()
    # cam.camera_tweak()
    while True:
        print(cam.record_project())
    """

    sys.setrecursionlimit(3000)
    th = Thymio.serial(port=os.getenv("COM_PORT"), refreshing_rate=0.1)
    # time.sleep(3)  # To make sure the Thymio has had time to connect

    # VelocityTuning(th)    # velocity tuning
    # MotionTuning(thymio=th, distance=60, angle=180.0) # motion tuning
    EventHandler(th)  # all the different scenarios are handled

    print("END OF MAIN!")


if __name__ == "__main__":
    main()
