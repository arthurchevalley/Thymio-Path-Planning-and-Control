import os
import cv2
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from ipywidgets import interact
from tqdm import tqdm

#  All color in order to compte the different mask
low_green = np.array([25, 35, 145])
up_green = np.array([80, 157, 200])
low_yellow = np.array([12, 55, 200])
up_yellow = np.array([27, 121, 255])
low_red = np.array([163, 111, 125])
up_red = np.array([[179, 193, 255]])
low_blue = np.array([98, 123, 88])
up_blue = np.array([119, 255, 215])

# Thymio outline
center_offset = np.array([5.5,5.5])
thymio_coords = np.array([[0,0], [11,0], [11,8.5], [10.2, 9.3],
                          [8, 10.4], [5.5,11], [3.1, 10.5],
                          [0.9, 9.4], [0, 8.5], [0,0]])-center_offset


LENGTH_case = 32
WIDTH_case = 29
FREE = 0
OCCUPIED = 1
RIGHT = 0
LEFT = 1
STRAIGHT = 2

LENGTH = 80
WIDTH = 72.5
LENGTH_G = 32
WIDTH_G = 29
gw = (LENGTH + 5)
gh = (WIDTH + 5)
zero_init = 0

# shapren parameter
alpha = 1.5  # Contrast control (1.0-3.0)
beta = 0  # Brightness control (0-100)




#----------------------------------------
#----------------------------------------
#     code for the notebook: VISION
#----------------------------------------
#----------------------------------------

def video_handle_for_demo():
    """
    Open the webcam video and save the current image in order to localize the thymio
    """
    frame = cv2.imread("vision.png")

    return frame

def record_project():
    """
    Main function in order to find the thymio on the map using the webcam
   
    :return:            Thymio pose and the image
    """
    # open the video and save the frame and return the fW,fH and the frame
    frame = video_handle_for_demo()

    # detect the blue square and resize the frame
    image = detect_and_rotate(frame)
    if image is None:
        return [-100, -100, 0]

    fW, fH, _ = image.shape

    # detect both yellow and green square for further angle and center computation
    x2g, y2g, xfg, yfg, frameg = frame_analysis_green(fW, fH, image)
    
    x2y, y2y, xfy, yfy, framey = frame_analysis_yellow(fW, fH, image)

    #  Correct the coordinate to have them in the true axis
    #  x2_ are the coordinate in grid referential
    #  xf_ are the coordinate in pixel of the resized image referential

    x2y = xfy
    x2g = xfg
    y2g = yfg
    y2y = yfy
    
    #  Compute the thymio center in grid's coordinate
    xc = (x2g + x2y) / 2
    yc = (y2g + y2y) / 2
    print("Picture of thymio with computed center represented by a green dot")
    cv2.circle(image,(round(xc),round(yc)),4,(255,255,0),-1)
    plt.figure()
    plt.imshow(image[:,:,::-1])
    plt.show()
    
    ratio = (gw / fH, gh / fW)

    xfg_temp = fW - (fH - yfg)
    yfg = xfg
    xfg = xfg_temp

    xfy_temp = fW - (fH - yfy)
    yfy = xfy
    xfy = xfy_temp
    
    #  Compute the angle thymio has
    angle = give_thymio_angle(image, xfy, yfy, xfg, yfg)

    x2g = x2g * ratio[0]
    x2y = x2y * ratio[0]
    y2g = y2g * ratio[1]
    y2y = y2y * ratio[1]

    # compute the center of the thymio & gives thymio angle
    xc = (x2g + x2y) / 2
    yc = (y2g + y2y) / 2

    # plot the image with the drawings and print the X,Y coordinate and the angle
    xc = xc - 2.5
    yc = yc - 2.5
    yc = 72.5 - yc
    
    
    return [xc, yc, angle], image

def frame_analysis_green(fW, fH, frame):
    """
    Find the lower part of the thmyio on the picture

    :param fW:      Width in pixel of the picture
    .param fH:      Height in pixel of the picture
    :param frame:   Image to analysis
    
    :return:        Coordinate in cm and pixel and the map
    """
    #  Compute the ratio to go from pixel coordinate to grid coordinate
    cam_grid_ratio = (gw / fW, gh / fH)

    #  Compute the green mask needed to find thymio
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, low_green, up_green)
    
    #  Find to contours of the square in order to compute the center of it
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    areas = [cv2.contourArea(c) for c in contours]
    
    #  If we don't find the square, return impossible value for the followin code
    #  to know no measurement were possible
    if len(areas) < 1:

        # Display the resulting frame
        x2, y2 = (-1, -1)
        xf, yf = (-1, -1)

    else:

        # Find the largest moving object in the image
        max_index = np.argmax(areas)
        cnt = contours[max_index]
        x, y, w, h = cv2.boundingRect(cnt)

        # Find the center of the green square
        xf = x + int(w / 2)
        yf = y + int(h / 2)

        #  Change from pixel coordinate to grid coordinate
        x2 = xf * cam_grid_ratio[0]
        y2 = gh - yf * cam_grid_ratio[1]

        frame = frame[:, :, ::-1]

    return x2, y2, xf, yf, frame

def frame_analysis_yellow( fW, fH, frame):
    """
    Find the upper part of the thmyio on the picture

    :param fW:      Width in pixel of the picture
    .param fH:      Height in pixel of the picture
    :param frame:   Image to analysis
    
    :return:        Coordinate in cm and pixel and the map
    """
    
    #  Compute the ratio to go from pixel coordinate to grid coordinate
    cam_grid_ratio = (gw / fW, gh / fH)

    #  Compute the yellow mask in order to find the uper square on the thymio
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, low_yellow, up_yellow)

    #  Find to contours of the square in order to compute the center of it
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    areas = [cv2.contourArea(c) for c in contours]

    #  If we don't find the square, return impossible value for the followin code
    #  to know no measurement were possible
    if len(areas) < 1:

        # Display the resulting frame
        x2, y2 = (-1, -1)
        xf, yf = (-1, -1)

    else:

        # Find the largest moving object in the image
        max_index = np.argmax(areas)
        cnt = contours[max_index]
        x, y, w, h = cv2.boundingRect(cnt)

        #  Change from pixel to grid coordinate
        xf = x + int(w / 2)
        yf = y + int(h / 2)

        x2 = xf * cam_grid_ratio[0]
        y2 = gh - yf * cam_grid_ratio[1]

        frame = frame[:, :, ::-1]

    return x2, y2, xf, yf, frame

def give_thymio_angle(image, xcy, ycy, xcg, ycg):
    """
    Compute the thymio angle

    :param xcy:     Yellow square center along X axis in pixel
    :param ycy:     Yellow square center along Y axis in pixel
    :param xcg:     Green square center along X axis in pixel
    :param ycg:     Green square center along Y axis in pixel
    :param image:   Actual image
    
    :return:        Thymio angle 
    """

    #  Find in which cadran thymio is
    y1 = int(ycy)
    y2 = int(ycg)
    x1 = int(xcy)
    x2 = int(xcg)

    if xcy > xcg:
        if ycg >= ycy:
            angle_rad = np.arctan2(np.abs(y1 - y2), np.abs(x1 - x2))
            angle = - np.rad2deg(angle_rad)

        else:
            angle_rad = np.arctan2(np.abs(y1 - y2), np.abs(x1 - x2))
            angle = np.rad2deg(angle_rad)

    else:
        if ycg >= ycy:
            angle_rad = np.arctan2(np.abs(y1 - y2), np.abs(x1 - x2))
            angle = np.rad2deg(angle_rad) - 180

        else:
            angle_rad = np.arctan2(np.abs(y1 - y2), np.abs(x1 - x2))
            angle = - np.rad2deg(angle_rad) + 180

    return angle

def detect_and_rotate(image):
    """
    Detect the blue square surronding the world in order to turn and resize the picture

    :param image:       Picture to analyse

    :return:            Resized image
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # hsv value for the various mask
    # computing of the blue mask to isolate the contours of the map
    mask_blue = cv2.inRange(hsv, low_blue, up_blue)
    print("Results of the blue mask")
    plt.figure()
    plt.imshow(mask_blue)
    plt.show()
    # find the outside blue contours of the map on the whole world
    contours, _ = cv2.findContours(mask_blue, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    # find the rectangle which includes the contours
    maxArea = 0
    best = None
    for contour in contours:
        area = cv2.contourArea(contour)

        if area > maxArea:
            maxArea = area
            best = contour
            
    #  If no map found
    if maxArea < 10:
        return None
    
    # Find the outside rectangle
    rect = cv2.minAreaRect(best)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    
    # crop image inside bounding box
    scale = 1
    W = rect[1][0]
    H = rect[1][1]

    # finding the box to rotate
    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)

    # finding the rotation angle between vertical and longest size of rectangle
    angle = rect[2]
    rotated = False
    if angle < -45:
        angle += 90
        rotated = True

    # rotation center and rotation matrix
    center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
    size = (int(scale * (x2 - x1)), int(scale * (y2 - y1)))
    M = cv2.getRotationMatrix2D((size[0] / 2, size[1] / 2), angle, 1.0)

    # cropping the image and rotating it
    cropped = cv2.getRectSubPix(image, size, center)
    cropped = cv2.warpAffine(cropped, M, size)
    croppedW = W if not rotated else H
    croppedH = H if not rotated else W

    corrected = cv2.getRectSubPix(cropped, (int(croppedW * scale), int(croppedH * scale)),
                                  (size[0] / 2, size[1] / 2))
    final_grid = np.array(corrected)
    return final_grid

def rotate(vis_map):
    """
    Rotate the map in order to put it in the right orientation

    :param vis_map:     picture to rotate

    :return:            The rotated and cropped map
    """
    low_green = np.array([36, 0, 0])
    up_green = np.array([86, 255, 255])
    # computing of the green mask to find the correct orientation of the map
    hsv = cv2.cvtColor(vis_map, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(hsv, low_green, up_green)
    grid_corner = np.array(mask_green)
    n_rows, n_cols = grid_corner.shape

    # turning the whole map until the orientation is good or let it in the initial position
    test_full_rot = 0
    rotation_marker_row = 0
    rotation_marker_col = n_cols - 1

    while grid_corner[rotation_marker_row][n_cols - 1] == 0:
        grid_corner = np.rot90(grid_corner)
        vis_map = np.rot90(vis_map)
        n_rows, n_cols = grid_corner.shape
        test_full_rot = test_full_rot + 1
        if test_full_rot == 4:
            break

    # final world map with goal, thymio and obstacle
    world = vis_map[1:n_rows - 1, 1:n_cols - 1]
    return world

def detect_object(world):
    """
    Fonction to detect the global obstacles and the goal

    :param world:       global map of the world from the camera after resize and rotation

    :return:            The goal coordinate and the obstacle grid
    """
    # create the map with only the obstucale to non-zero
    world_hsv = cv2.cvtColor(world, cv2.COLOR_BGR2HSV)
    mask_red = cv2.inRange(world_hsv, low_red, up_red)
    occupancy_grid = np.array(mask_red)
    world_rows, world_cols, _ = world.shape

    #  create the mask in order to find the goal
    world_hsv = cv2.cvtColor(world, cv2.COLOR_BGR2HSV)
    mask_goal = cv2.inRange(world_hsv, low_blue, up_blue)
    goal_x, goal_y = (15, 15)  # goal by default

    # look for the obstacle and increase there size
    for i in range(world_rows):
        for j in range(world_cols):
            occupancy_grid[i][j] = int(occupancy_grid[i][j] / 255)
            if mask_goal[i][j] > 200:
                goal_x, goal_y = (i, j)
    object_grid = [[goal_x, goal_y]]
    return object_grid, occupancy_grid

def vision(image):
    """
   Main function handling the pictures capture and preparation before analysis

   :param image:       Picture took with the webcam

   :return:             The goal coordinate, the obstacle map and the full map
   """
    vis_map = resize(image, alpha, beta)
    print("Resized map from the blue mask")
    
    world = rotate(vis_map)
    
    plt.figure()
    plt.imshow(world[:,:,::-1])
    plt.show()
    object_grid, occupancy_grid = detect_object(world)
    print("Result of the red mask")
    plt.figure()
    plt.imshow(occupancy_grid)
    plt.show()
    return object_grid, occupancy_grid, world

def resize(final_grid, alpha, beta):
    """
    Resize to the desired grid size the camera picture

    :param final_grid:      The picture to resize
    :param alpha:           Sharpen's contrast control to resize the grid
    :param beta:            Sharpen's brightness control to resize the grid

    :return:                Resized map
    """
    #init the filter before resize
    adjusted = cv2.convertScaleAbs(final_grid, alpha, beta)
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpen = cv2.filter2D(final_grid, -1, sharpen_kernel)

    # resize the map to the wanted size
    map_w_border_row = WIDTH_G + 2
    map_w_border_col = LENGTH_G + 2
    vis_map = cv2.resize(sharpen, (map_w_border_col, map_w_border_row), cv2.INTER_AREA)
    return vis_map

def localize(image):
    """
    Main function handling the image analysis and thymio localisation with the camer

    :param image:       Image of the state

    :return:            the grid with increased obstacles and the goal coordinate
    """

    #  Call the vision function in order to have the grid with the obstacle and the goal coordinate
    object_grid, occupancy_grid, world = vision(image)

    #  Correction of the goal coordinate in order to fit the A* coordinate
    goal_x = object_grid[0][1]
    goal_y = WIDTH_G - object_grid[0][0]
    goal_coor = (goal_x, goal_y)


    return occupancy_grid, goal_coor



#-------------------------------------------------
#-------------------------------------------------
#     code for the notebook: GLOBAL NAVIGATION
#-------------------------------------------------
#-------------------------------------------------

def increased_obstacles_map(occupancy_grid):
    """
    Increase the size of the obstacles by the robot's radius
    :param occupancy_grid: 2D matrix containing the values of each cell in the map
    :return: 2D matrix containing the values of each cell in the map with increased obstacles
    """
    
    nb_rows = len(occupancy_grid)
    nb_cols = len(occupancy_grid[0])
    increased_occupancy_grid = np.zeros([nb_rows+6, nb_cols+6])
    
    for i in range(nb_rows):
        for j in range(nb_cols):
            
            if occupancy_grid[i,j] == OCCUPIED:
                increased_occupancy_grid[i:i+7,j:j+7] = np.ones([7,7])
                
    final_occupancy_grid = increased_occupancy_grid[3:(LENGTH_case + 3),3:(WIDTH_case + 3)]
    return final_occupancy_grid



def _get_movements_8n():
    """Get all possible 8-connectivity movements.
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
        current=cameFrom[current]
    return total_path

import math

def A_Star(start, goal, final_occupancy_grid):
    """
    Execution of the A* algorithm for 2D occupancy grid. Finds a path from start to goal.
    h is the heuristic function. h(n) estimates the cost to reach goal from node n.
    :param start: start node (x, y)
    :param goal: goal node (x, y)
    :param occupancy_grid: the grid map
    :return: a tuple that contains: the resulting path in data array indices
    """
    x,y = np.mgrid[0:45:1, 0:42:1]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x; pos[:, :, 1] = y
    pos = np.reshape(pos, (x.shape[0]*x.shape[1], 2))
    coords = list([(int(x[0]), int(x[1])) for x in pos])

    # Define the heuristic:
    # h: dictionary containing the distance to goal ignoring obstacles for all coordinates in the grid (heuristic function)
    h = np.linalg.norm(pos - goal, axis = 1)
    h = dict(zip(coords, h))

    # Check if the start and goal are within the boundaries of the map
    for point in [start, goal]:
       
        if point[0]<0 and point[0]>=final_occupancy_grid.shape[0]:
            raise Exception('Start node/goal node is not contained in the map')
  
        if point[1]<0 and point[1]>=final_occupancy_grid.shape[1]:
            raise Exception('Start node/goal node is not contained in the map')
    
    # check if start and goal nodes correspond to free spaces
    if final_occupancy_grid[start[0], start[1]]:
        raise Exception('Start node is not traversable')

    if final_occupancy_grid[goal[0], goal[1]]:
        raise Exception('Goal node is not traversable')
    
    # get the possible movements
    movements = _get_movements_8n()
    
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
        
        #the node in openSet having the lowest fScore[] value
        fScore_openSet = {key:val for (key,val) in fScore.items() if key in openSet}
        current = min(fScore_openSet, key=fScore_openSet.get)
        del fScore_openSet
        
        #If the goal is reached, reconstruct and return the obtained path
        if current == goal:
            return reconstruct_path(cameFrom, current)
        
        openSet.remove(current)
        closedSet.append(current)
        
        #for each neighbor of current:
        for dx, dy, deltacost in movements:
            
            neighbor = (current[0]+dx, current[1]+dy)
            
            # if the node is not in the map, skip
            if (neighbor[0] >= final_occupancy_grid.shape[0]) or (neighbor[1] >= final_occupancy_grid.shape[1]) or (neighbor[0] < 0) or (neighbor[1] < 0):
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
    return []


def display_global_path(start, goal, path, occupancy_grid):
    """
    Display the path found with the A star algorithm from the start to the goal node.
    :param start: start node (x, y)
    :param goal: goal node (x, y)
    :param occupancy_grid: the grid map
    """
    # Displaying the map
    fig_astar, ax_astar = display_map(occupancy_grid)

    # Plot the best path found and the list of visited nodes
    ax_astar.plot(path[0], path[1], marker="o", color = 'blue');
    ax_astar.scatter(start[0], start[1], marker="o", color = 'green', s=200);
    ax_astar.scatter(goal[0], goal[1], marker="o", color = 'purple', s=200);
    # ax.set_ylim(ax.get_ylim()[::-1])


def display_map(grid):
   """
   Display a map (occupancy grid)
   :param grid: 2D matrix containing the values of each cell in the map
   :return: the fig, ax objects.
   """
   fig, ax = plt.subplots(figsize=(7,7))
   
   major_ticks_x = np.arange(0, LENGTH_case+1, 5)
   minor_ticks_x = np.arange(0, LENGTH_case+1, 1)
   major_ticks_y = np.arange(0, WIDTH_case+1, 5)
   minor_ticks_y = np.arange(0, WIDTH_case+1, 1)
   ax.set_xticks(major_ticks_x)
   ax.set_xticks(minor_ticks_x, minor=True)
   ax.set_yticks(major_ticks_y)
   ax.set_yticks(minor_ticks_y, minor=True)
   ax.grid(which='minor', alpha=0.2)
   ax.grid(which='major', alpha=0.5)
   ax.set_ylim([0,WIDTH_case])
   ax.set_xlim([0,LENGTH_case])
   ax.grid(True)

   # Select the colors with which to display obstacles and free cells
   cmap = colors.ListedColormap(['white', 'red'])
       
   # Displaying the map
   ax.imshow(grid, cmap=cmap)
   plt.title("Map : free cells in white, occupied cells in red");
       
   return fig,ax

#----------------------------------------
#----------------------------------------
# code for the notebook: MOTION CONTROL
#----------------------------------------
#----------------------------------------

def path_to_command_thymio(path):
    """
    Find a list of commands (left, right, straight) from the path
    :param path: a list containing, a list of the x coordinates and a list of the y coordinates of the path
    :return: a boolean telling if the robot turns to the LEFT, RIGHT or goes STRAIGHT; return the path given as input without the first element
    """

    current_x = path[0][0]
    current_y = path[1][0]
    
    next_x = path[0][1]
    next_y = path[1][1]
    
    # next-prev
    delta_x = path[0][1] - path[0][0]
    delta_y = path[1][1] - path[1][0]
    
    
    # delat_x = 0 and delta_y = -/+ 1 (or delat_x = -/+ 1 and delta_y = 0): go straight
    turn = STRAIGHT
    
    # delat_x = -1 and delta_y = 1 (or delat_x = 1 and delta_y = -1): turn to the right
    if delta_x*delta_y < 0:
        turn = RIGHT
    
    # delat_x = -1 and delta_y = -1 (or delat_x = 1 and delta_y = 1): turn to the left
    if delta_x*delta_y == 1:
        turn = LEFT
    
    new_path = np.array([path[0][1:],path[1][1:]])
    
    return turn, new_path


def full_path_to_points(path):
    """
    Find the corners of the path
    :param path: a list containing, a list of the x coordinates and a list of the y coordinates of the path
    :return:a list containing, a list of the x coordinates and a list of the y coordinates of the corners of the path
    """
    
    points_x = [path[0][0]]
    points_y = [path[1][0]]
    

    new_path = path
    prev_turn, new_path = path_to_command_thymio(new_path)
    
    for i in range(len(new_path[0])-1):
        
        new_turn, new_path = path_to_command_thymio(new_path)
        
        if new_turn!= prev_turn:
            points_x.append(path[0][i+1])
            points_y.append(path[1][i+1])
        
        prev_turn = new_turn
    
    points_x.append(path[0][-1])
    points_y.append(path[1][-1])
    points = [points_x, points_y]
    
    return points


#----------------------------------------
#----------------------------------------
# code for the notebook: LOCAL AVOIDANCE
#----------------------------------------
#----------------------------------------


def get_frames_and_titles(video_label):
    """
    Take frames from a video
    :param video_label: name of the video, from which we take frames
    :return: frames (a list of frames form the video), titles (a list of titles (containing integers) corresponding to the frames number9
    """
    
    frames = []
    titles = []
    cap = cv2.VideoCapture(video_label)
    frameRate = cap.get(5) # frame rate
    numero=29
    index=0
    while(cap.isOpened()):
        frameId = cap.get(1) # current frame number
        ret, frame = cap.read()
        # Apply template Matching
        if (ret != True):
            break
        if (frameId % math.floor(frameRate) == 0):
            frames.append(frame)
            index=index
            titles.append("Frame {}".format(int(index/30)))

        if numero > 30:
            break
        index=index+1
    cap.release()
    
    return frames, titles


def browse_images(images, titles = None):
    """
    Create the interactive view of the frames (browser)
    :param images: a list of frames
    :param titles: a list of titles (integers)
    """

    if titles == None:
        titles = [i for i in range(len(images))]
        
    n = len(images)
    def view_image(i):
        plt.imshow(images[i][:,:,::-1], cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title(titles[i], y=-0.5)
        plt.show()
    interact(view_image, i=(1,n-1))
    
    
#----------------------------------------
#----------------------------------------
# code for the notebook: KALMAN FILTER
#----------------------------------------
#----------------------------------------

# model params
Ts = 0.1
qx = 0.2
qy = 0.2
qt = 0.4
k_delta_sr = 0.8
k_delta_sl = 0.8

b = 0.095
H = np.array([[1, 0, 0], [0, 1, 0],[0, 0, 1]])
Q = np.array([[qx, 0, 0], [0, qy, 0], [0, 0, qt]])
R = np.array([[k_delta_sr, 0], [0, k_delta_sl]])

def jacobianF_x(theta, delta_s, delta_theta):
    """
    Compute the partial derivative of the motion model with respect to the state vector x, evaluated at the current state x and input u
    
    :param theta: current orientation of the robot
    :param delta_s: mean of the travelled distance of the right wheel and the left wheel
    :param delta_theta: angle increment based on the travelled distance of the right wheel and the left wheel, and the distance between the wheels
    
    :return: a matrix (np.array) containing the partial derivative evaluated at the current state and input u
    """
    
    Fx = np.array([[1, 0, -delta_s*np.sin(theta + delta_theta/2)], [0, 1, delta_s*np.cos(theta + delta_theta/2)], [0, 0, 1]])
    return Fx

def jacobianF_u(theta, delta_s, delta_theta):
    """
    Compute the partial derivative of the motion model with respect to the input vector u, evaluated at the current state x and input u
    
    :param theta: current orientation of the robot
    :param delta_s: mean of the travelled distance of the right wheel and the left wheel
    :param delta_theta: angle increment based on the travelled distance of the right wheel and the left wheel, and the distance between the wheels
    
    :return: a matrix (np.array) containing the partial derivative evaluated at the current state x and input u
    """

    Fu = np.array([[1/2*np.cos(theta + delta_theta/2) - delta_s/(2*b)*np.sin(theta + delta_theta/2), 1/2*np.cos(theta + delta_theta/2) + delta_s/(2*b)*np.sin(theta + delta_theta/2)], [1/2*np.sin(theta + delta_theta/2) + delta_s/(2*b)*np.cos(theta + delta_theta/2), 1/2*np.sin(theta + delta_theta/2) - delta_s/(2*b)*np.cos(theta + delta_theta/2)], [1/b , -1/b]])
    return Fu


def prediction_only(state_est_prev, cov_est_prev, delta_sr, delta_sl):
    """
    Estimates the current state using only the previous state
    
    param delta_sr: travelled distance for the right wheel (in meters)
    param delta_sl: travelled distance for the left wheel (in meters)
    param state_est_prev: previous state a posteriori estimation
    param cov_est_prev: previous state a posteriori covariance

    return state_est_a_priori: new a priori state estimation
    return cov_est: new a priori state covariance
    """
    
    theta = state_est_prev[2,0]
    delta_s = (delta_sr + delta_sl)/2
    delta_theta = (delta_sr - delta_sl)/b
    
    Fx = jacobianF_x(theta, delta_s, delta_theta)
    Fu = jacobianF_u(theta, delta_s, delta_theta)
    
    ## Prediciton step
    # estimated mean of the state
    state_est_a_priori = state_est_prev + np.array([[delta_s*np.cos(theta + delta_theta/2)],[delta_s*np.sin(theta + delta_theta/2)],[delta_theta]])

    
    # Estimated covariance of the state
    cov_est_a_priori = np.dot(Fx, np.dot(cov_est_prev, Fx.T)) + np.dot(Fu, np.dot(R, Fu.T))
    
    return state_est_a_priori, cov_est_a_priori


def kalman_filter_with_indices(i, z, state_est_prev, cov_est_prev, delta_sr, delta_sl):
   """
   Estimates the current state using input sensor data and the previous state
   
   param z: array representing the measurement (x,y,theta) (coming from the vision sensor)
   param delta_sr: travelled distance for the right wheel (in meters)
   param delta_sl: travelled distance for the left wheel (in meters)
   param state_est_prev: previous state a posteriori estimation
   param cov_est_prev: previous state a posteriori covariance

   return state_est: new a posteriori state estimation
   return cov_est: new a posteriori state covariance
   """
   
   theta = state_est_prev[2,0]
   delta_s = (delta_sr + delta_sl)/2
   delta_theta = (delta_sr - delta_sl)/b
   
   Fx = jacobianF_x(theta, delta_s, delta_theta)
   Fu = jacobianF_u(theta, delta_s, delta_theta)
   
   ## Prediciton step
   # estimated mean of the state
   state_est_a_priori = state_est_prev + np.array([[delta_s*np.cos(theta + delta_theta/2)],[delta_s*np.sin(theta + delta_theta/2)],[delta_theta]])

   
   # Estimated covariance of the state
   cov_est_a_priori = np.dot(Fx, np.dot(cov_est_prev, Fx.T)) + np.dot(Fu, np.dot(R, Fu.T))
   
   if i == 2 or i == 4:
       ## Update step
       # innovation / measurement residual
       i = z - state_est_a_priori;

       # Kalman gain (tells how much the predictions should be corrected based on the measurements)
       K = np.dot(cov_est_a_priori, np.linalg.inv(cov_est_a_priori + Q));
   
       # a posteriori estimate
       state_est = state_est_a_priori + np.dot(K,i);
       cov_est = cov_est_a_priori - np.dot(K,cov_est_a_priori);

   else:
       state_est = state_est_a_priori
       cov_est = cov_est_a_priori
           
   return state_est, cov_est
   
   

def kalman_filter(z, state_est_prev, cov_est_prev, delta_sr, delta_sl):
    """
    Estimates the current state using input sensor data and the previous state

    param z: array representing the measurement (x,y,theta) (coming from the vision sensor)
    param delta_sr: travelled distance for the right wheel (in meters)
    param delta_sl: travelled distance for the left wheel (in meters)
    param state_est_prev: previous state a posteriori estimation
    param cov_est_prev: previous state a posteriori covariance

    return state_est: new a posteriori state estimation
    return cov_est: new a posteriori state covariance
    """

    theta = state_est_prev[2,0]
    delta_s = (delta_sr + delta_sl)/2
    delta_theta = (delta_sr - delta_sl)/b

    Fx = jacobianF_x(theta, delta_s, delta_theta)
    Fu = jacobianF_u(theta, delta_s, delta_theta)

    ## Prediciton step
    # estimated mean of the state
    state_est_a_priori = state_est_prev + np.array([[delta_s*np.cos(theta + delta_theta/2)],[delta_s*np.sin(theta + delta_theta/2)],[delta_theta]])


    # Estimated covariance of the state
    cov_est_a_priori = np.dot(Fx, np.dot(cov_est_prev, Fx.T)) + np.dot(Fu, np.dot(R, Fu.T))

    ## Update step
    # innovation / measurement residual
    i = z - state_est_a_priori;

    # Kalman gain (tells how much the predictions should be corrected based on the measurements)
    K = np.dot(cov_est_a_priori, np.linalg.inv(cov_est_a_priori + Q));

    # a posteriori estimate
    state_est = state_est_a_priori + np.dot(K,i);
    cov_est = cov_est_a_priori - np.dot(K,cov_est_a_priori);

            
    return state_est, cov_est
    
    

def plot_covariance_ellipse(state_est, cov_est):
    """
    Plot the position covariance matrix of the current state as an ellipse whose axis are the square root of its eigenvalues
    :param state_est: state estimation (np.array of size 3 x 1)
    :param cov_est: covariance estimation (np.array of size 3 x 3)
    
    :return: list of the coordinate x of the ellipse, list of the coordinate y of the ellipse
    """
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
    
    
def get_param_scenario3():
    """
    Initialize the parameters of scenario 3

    :return: list of travelled distance for the right wheel, list of travelled distance for the left wheel, list of all measurements
    """
    nb_carre_x = 42
    nb_carre_y = 45
    largeur_x = 0.825
    largeur_y = 0.645
    z1 = np.array([[11], [32], [17.26]])
    z2 = np.array([[12], [32], [18.43]])
    z3 = np.array([[13], [32], [23.2]])
    z4 = np.array([[15], [31], [23.4]])
    z5 = np.array([[16], [30], [23.29]])
    z6 = np.array([[18], [29], [22.28]])
    z7 = np.array([[19], [28], [35.79]])
    z8 = np.array([[20], [27], [36.87]])
    z9 = np.array([[21], [26], [33.92]])
    z10 = np.array([[23], [24], [38.11]])
    z11 = np.array([[24], [23], [37.76]])
    z12 = np.array([[25], [22], [45.6]])
    z13 = np.array([[26], [20], [56.4]])
    z14 = np.array([[27], [18], [55.2]])
    z15 = np.array([[28], [16], [57.53]])
    z16 = np.array([[29], [14], [58.64]])
    z17 = np.array([[30], [13], [66.04]])
    z18 = np.array([[30], [11], [70.02]])
    z19 = np.array([[31], [9], [69.64]])
    z20 = np.array([[31], [6], [68.82]])

    meas = [z1, z2, z3, z4, z5, z6, z7, z8, z9, z10, z11, z12, z13, z14, z15, z16, z17, z18, z19, z20]

    z = [np.array([[x[0][0]*largeur_x/nb_carre_x],[x[1][0]*largeur_y/nb_carre_y],[-x[2][0]*np.pi/180]]) for x in meas]

    # We take a measure every second
    thymio_speed_to_mms = 0.4753
    Ts = 1

    Thymio_speed_left = [73, 92, 94,  102, 100, 89,  92,  103, 97, 100, 100, 104, 95, 97, 103, 101, 100, 101, 92, 98]
    Thymio_speed_right = [67, 96, 102, 89, 105, -45, 102, 101, 104, 71, 100, 93, 94, 105, 96, -45, 95, 103, 88, 94]
    delta_sr_test = [x*Ts/thymio_speed_to_mms/1000 for x in Thymio_speed_right]
    delta_sl_test = [x*Ts/thymio_speed_to_mms/1000 for x in Thymio_speed_left]


    return delta_sr_test, delta_sl_test, z
    
    
def rotate_thymio(angle, coords):
    """
    Rotates the coordinates of a matrix by the desired angle
    :param angle: angle in radians by which we want to rotate
    :return: numpy.array() that contains rotated coordinates
    """
    R = np.array(((np.cos(angle), -np.sin(angle)),
                  (np.sin(angle),  np.cos(angle))))

    return R.dot(coords.transpose()).transpose()

