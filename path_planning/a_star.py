import numpy as np
import math

LENGTH = 32
WIDTH = 29


def reconstruct_path(cameFrom, current):
    """
    Recurrently reconstructs the path from start node to the current node
    :param cameFrom: map (dictionary) containing for each node n the node immediately
                     preceding it on the cheapest path from start to n
                     currently known.
    :param current: current node (x, y)
    :return: list of nodes from start to current node
    """
    total_path = [current]
    while current in cameFrom.keys():
        # Add where the current node came from to the start of the list (add cameFrom[current] at the 0th index)
        total_path.insert(0, cameFrom[current])
        current = cameFrom[current]
    return total_path


def get_movements_8n():
    """
    Get all possible 8-connectivity movements.
    - up
    - down
    - left
    - right
    - first diagonal (up-right)
    - second diagonal (up-left)
    - third diagonal (down-right)
    - fourth diagonal (down-left)
    :return: list of movements with cost [(dx, dy, movement_cost)]
    """
    s2 = math.sqrt(2)
    return [(1, 0, 1.0),
            (0, 1, 1.0),
            (-1, 0, 1.0),
            (0, -1, 1.0),
            (1, 1, s2),
            (-1, 1, s2),
            (-1, -1, s2),
            (1, -1, s2)]


def A_Star(start, goal, final_occupancy_grid):
    """
    Execution of the A* algorithm for 2D occupancy grid. Finds a path from start to goal.
    h is the heuristic function. h(n) estimates the cost to reach goal from node n.
    :param start: start node (x, y)
    :param goal: goal node (x, y)
    :param occupancy_grid: the grid map
    :return: a tuple that contains: (the resulting path in meters, the resulting path in data array indices)
    """
    x, y = np.mgrid[0:LENGTH:1, 0:WIDTH:1]
    pos = np.empty(x.shape + (2,))
    # x.shape = (LENGTH,WIDTH)
    # x.shape + (2,) = (LENGTH,WIDTH,2)
    pos[:, :, 0] = x
    pos[:, :, 1] = y
    # pos.shape = (1890, 2)
    pos = np.reshape(pos, (x.shape[0] * x.shape[1], 2))
    coords = list([(int(x[0]), int(x[1])) for x in pos])
    # Define the heuristic:
    # h: dictionary containing the distance to goal ignoring obstacles for all coordinates in the grid (heuristic function)
    h = np.linalg.norm(pos - goal, axis=1)
    # If axis is an integer, it specifies the axis of x along which to compute the vector norms
    # axis = 1: h.shape  = 1890
    # axis = 0: h.shape  = 2
    h = dict(zip(coords, h))

    # Check if the start and goal are within the boundaries of the map
    for point in [start, goal]:

        if point[0] < 0 and point[0] >= final_occupancy_grid.shape[0]:
            raise Exception('Start node/goal node is not contained in the map')

        if point[1] < 0 and point[1] >= final_occupancy_grid.shape[1]:
            raise Exception('Start node/goal node is not contained in the map')

    # check if start and goal nodes correspond to free spaces
    if final_occupancy_grid[start[0], start[1]]:
        raise Exception('Start node is not traversable')

    if final_occupancy_grid[goal[0], goal[1]]:
        raise Exception('Goal node is not traversable')

    # get the possible movements
    movements = get_movements_8n()

    # The set of visited nodes that need to be (re-)expanded, i.e. for which the neighbors need to be explored
    # Initially, only the start node is known.
    openSet = [start]

    # The set of visited nodes that no longer need to be expanded.
    closedSet = []

    # For node n, cameFrom[n] is the node immediately preceding it on the cheapest path from start to n currently known.
    cameFrom = dict()

    # For node n, gScore[n] is the cost of the cheapest path from start to n currently known.
    gScore = dict(zip(coords, [np.inf for x in range(len(coords))]))
    gScore[start] = 0

    # For node n, fScore[n] := gScore[n] + h(n). map with default value of Infinity
    fScore = dict(zip(coords, [np.inf for x in range(len(coords))]))
    fScore[start] = h[start]

    # while there are still elements to investigate
    while openSet != []:

        # the node in openSet having the lowest fScore[] value
        fScore_openSet = {key: val for (key, val) in fScore.items() if key in openSet}
        current = min(fScore_openSet, key=fScore_openSet.get)
        del fScore_openSet

        # If the goal is reached, reconstruct and return the obtained path
        if current == goal:
            # print("Path", closedSet)
            return reconstruct_path(cameFrom, current)

        openSet.remove(current)
        closedSet.append(current)

        # for each neighbor of current:
        for dx, dy, deltacost in movements:

            neighbor = (current[0] + dx, current[1] + dy)

            # if the node is not in the map, skip
            if (neighbor[0] >= final_occupancy_grid.shape[0]) or (neighbor[1] >= final_occupancy_grid.shape[1]) or (
                    neighbor[0] < 0) or (neighbor[1] < 0):
                continue

            # if the node is occupied, skip
            if (final_occupancy_grid[neighbor[0], neighbor[1]]):
                continue

            # if the has already been visited, skip
            if (neighbor in closedSet):
                continue
            # d(current,neighbor) is the weight of the edge from current to neighbor
            # tentative_gScore is the distance from start to the neighbor through current
            tentative_gScore = gScore[current] + deltacost

            if neighbor not in openSet:
                openSet.append(neighbor)

            if tentative_gScore < gScore[neighbor]:
                # This path to neighbor is better than any previous one. Record it!
                cameFrom[neighbor] = current
                gScore[neighbor] = tentative_gScore
                fScore[neighbor] = gScore[neighbor] + h[neighbor]

    # Open set is empty but goal was never reached
    print("No path found to goal")
    return [], closedSet
