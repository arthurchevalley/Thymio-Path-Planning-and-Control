import threading
import time

from src.displacement.movement import stop, move, advance, rotate
from src.sensors.state import SensorHandler
from src.thymio.Thymio import Thymio
from dotenv import load_dotenv
from threading import Timer

import matplotlib.pyplot as plt

load_dotenv()


class MotionTuning:
    """
    Tune the robot to go in a straight line.
    Be careful, make it run for more than one lien because apparently there are some error initially.

    Example:
    InitTuning(thymio=th, distance=15.0, angle=180.0)
    """

    def __init__(self, thymio: Thymio, interval_check=0.1, interval_sleep=0.1, distance=60.0, angle=180.0):
        self.thymio = thymio
        self.interval_check = interval_check
        self.interval_sleep = interval_sleep
        self.distance = distance
        self.angle = angle
        self.timer_advance = Timer(interval=interval_sleep, function=stop)
        self.timer_rotate = Timer(interval=interval_sleep, function=stop)
        stop(self.thymio, verbose=True)
        self.__tune_handler()

    def __tune_handler(self):
        advance(thymio=self.thymio, distance=self.distance, verbose=True)
        rotate(thymio=self.thymio, angle=self.angle, verbose=True)
        self.__tune_handler()


class VelocityTuning:
    """
    """

    def __init__(self, thymio: Thymio, interval_sleep=0.1):
        self.thymio = thymio
        self.sensor_handler = SensorHandler(self.thymio)
        self.interval_sleep = interval_sleep
        self.time = time.time()
        self.left = []
        self.right = []
        self.__forward()

    def __record(self):
        sensor = self.sensor_handler.all_raw()
        print(sensor)
        self.left.append(sensor['ground'][0])
        self.right.append(sensor['ground'][1])

    def __forward(self):
        move(self.thymio, l_speed_ratio=0.8, r_speed_ratio=0.8, verbose=True)

        while time.time() - self.time < 20:
            threading.Thread(target=self.__record).start()
            time.sleep(0.1)

        stop(self.thymio)
        self.__plot()

    def __plot(self):
        plt.plot(self.left, label="left sensor")
        plt.plot(self.right, label="right sensor")
        # plt.plot(l_peaks, [l_sensor[idx] for idx in l_peaks], "o", label="left sensor peaks")
        plt.xlabel("Time step 0.1[s]")
        plt.ylabel("Ground sensor measurement")
        plt.legend()
        plt.show()
        print("now find the time for, knowing that one interval is 50mm")
        print("time interval divide / 50[mm] / 80[speed]")
        print("nb_pikes*50/0.1/time for nb_of_spikes")


class SensorTuning:
    """
    """

    def __init__(self, thymio: Thymio):
        self.thymio = thymio
