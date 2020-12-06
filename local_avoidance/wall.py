from src.path_planning.localization import test_saw_black
from src.thymio.Thymio import Thymio
from src.displacement.movement import move


def test_saw_wall(thymio: Thymio, wall_threshold, verbose=False) -> bool:
    """
    Tests whether one of the proximity sensors saw a wall

    :param thymio:          The file location of the spreadsheet
    :param wall_threshold:  threshold starting which it is considered that the sensor saw a wall
    :param verbose:         whether to print status messages or not
    :return: bool           existence of wall or not
    """

    if any([x > wall_threshold for x in thymio['prox.horizontal'][:-2]]):
        if verbose:
            print("\t\t Saw a wall")
        return True

    return False


def wall_following(thymio: Thymio, motor_speed: int = 20, wall_threshold: int = 500, white_threshold: int = 200,
                   verbose: bool = False):
    """
    Wall following behaviour of the FSM

    :param thymio:          The file location of the spreadsheet
    :param motor_speed: the Thymio's motor speed
    :param wall_threshold: threshold starting which it is considered that the sensor saw a wall
    :param white_threshold: threshold starting which it is considered that the ground sensor saw white
    :param verbose: whether to print status messages or not
    """

    if verbose:
        print("Starting wall following behaviour")
    saw_black = False

    if verbose:
        print("\t Moving forward")
    move(thymio, l_speed=motor_speed, r_speed=motor_speed)

    prev_state = "forward"

    while not saw_black:

        if test_saw_wall(thymio, wall_threshold, verbose=False):
            if prev_state == "forward":
                if verbose:
                    print("\tSaw wall, turning clockwise")
                move(thymio, l_speed=motor_speed, r_speed=-motor_speed)
                prev_state = "turning"

        else:
            if prev_state == "turning":
                if verbose:
                    print("\t Moving forward")
                move(thymio, l_speed=motor_speed, r_speed=motor_speed)
                prev_state = "forward"

        if test_saw_black(thymio, white_threshold, verbose):
            saw_black = True

    return
