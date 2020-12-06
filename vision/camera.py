import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()


class Colors:

    def __init__(self):
        self.low_green = np.array([25, 35, 145])
        self.up_green = np.array([80, 157, 200])
        self.low_yellow = np.array([12, 55, 200])
        self.up_yellow = np.array([27, 121, 255])
        self.low_red = np.array([163, 111, 125])
        self.up_red = np.array([[179, 193, 255]])
        self.low_blue = np.array([98, 123, 88])
        self.up_blue = np.array([119, 255, 215])


class Camera:
    def __init__(self):
        self.LENGTH = 80
        self.WIDTH = 72.5
        self.gw = (self.LENGTH + 5)
        self.gh = (self.WIDTH + 5)
        self.colors = Colors()
        self.cap = None

    def open_camera(self):
        filename = int(os.getenv("CAMERA_PORT"))
        self.cap = cv2.VideoCapture(filename)

    def close_camera(self):
        self.cap.release()

    def video_handle(self):
        """
        Open the webcam video and save the current image in order to localize the thymio
        :param filename:        video path to open
        """
        #  Take the wanted frame
        _, frame = self.cap.read()

        fH, fW, _ = frame.shape

        return fW, fH, frame

    def record_project(self):
        """
        Main function in order to find the thymio on the map using the webcam
        """
        # open the video and save the frame and return the fW,fH and the frame
        fW, fH, frame = self.video_handle()

        # detect the blue square and resize the frame
        image = self.detect_and_rotate(frame)

        if image is None:
            return [-100, -100, 0]

        fW, fH, _ = image.shape

        # detect both yellow and green square for further angle and center computation
        x2g, y2g, xfg, yfg, frameg = self.frame_analysis_green(fW, fH, image)
        x2y, y2y, xfy, yfy, framey = self.frame_analysis_yellow(fW, fH, image)

        x2y = xfy
        x2g = xfg
        y2g = yfg
        y2y = yfy
        ratio = (self.gw / fH, self.gh / fW)

        xfg_temp = fW - (fH - yfg)
        yfg = xfg
        xfg = xfg_temp

        xfy_temp = fW - (fH - yfy)
        yfy = xfy
        xfy = xfy_temp

        angle = self.give_thymio_angle(image, xfy, yfy, xfg, yfg)

        x2g = x2g * ratio[0]
        x2y = x2y * ratio[0]
        y2g = y2g * ratio[1]
        y2y = y2y * ratio[1]

        # compute the center of the thymio & gives thymio angle
        xc = (x2g + x2y) / 2
        yc = (y2g + y2y) / 2
        #  Correct the value to be in the grid
        xc = xc - 2.5
        yc = yc - 2.5
        yc = 72.5 - yc
        return [xc, yc, angle]

    def frame_analysis_green(self, fW, fH, frame):
        """
        Find the lower part of the thmyio on the picture
        :param fW:      Width in pixel of the picture
        :param fH:      Height in pixel of the picture
        :param frame:   Image to analysis
        """

        cam_grid_ratio = (self.gw / fW, self.gh / fH)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, self.colors.low_green, self.colors.up_green)
        res = cv2.bitwise_and(frame, frame, mask=mask)

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        areas = [cv2.contourArea(c) for c in contours]
        # print("length: ", len(areas))
        if len(areas) < 1:

            # Display the resulting frame
            x2, y2 = (-1, -1)
            xf, yf = (-1, -1)

        else:

            # Find the largest moving object in the image
            max_index = np.argmax(areas)

            cnt = contours[max_index]

            x, y, w, h = cv2.boundingRect(cnt)
            # cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)

            # Draw circle in the center of the bounding box
            xf = x + int(w / 2)
            yf = y + int(h / 2)
            # cv2.circle(frame,(xf,yf),4,(255,255,0),-1)

            x2 = xf * cam_grid_ratio[0]
            y2 = self.gh - yf * cam_grid_ratio[1]

            frame = frame[:, :, ::-1]

        return x2, y2, xf, yf, frame

    def frame_analysis_yellow(self, fW, fH, frame):
        """
        Find the upper part of the thmyio on the picture
        :param fW:      Width in pixel of the picture
        :param fH:      Height in pixel of the picture
        :param frame:   Image to analysis
        """
        cam_grid_ratio = (self.gw / fW, self.gh / fH)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.colors.low_yellow, self.colors.up_yellow)
        res = cv2.bitwise_and(frame, frame, mask=mask)

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        areas = [cv2.contourArea(c) for c in contours]
        if len(areas) < 1:

            # Display the resulting frame
            x2, y2 = (-1, -1)
            xf, yf = (-1, -1)

        else:

            # Find the largest moving object in the image
            max_index = np.argmax(areas)

            cnt = contours[max_index]
            x, y, w, h = cv2.boundingRect(cnt)

            # Draw circle in the center of the bounding box
            xf = x + int(w / 2)
            yf = y + int(h / 2)
            cv2.circle(frame, (xf, yf), 4, (255, 255, 0), -1)

            x2 = xf * cam_grid_ratio[0]
            y2 = self.gh - yf * cam_grid_ratio[1]

            frame = frame[:, :, ::-1]

        return x2, y2, xf, yf, frame

    def give_thymio_angle(self, image, xcy, ycy, xcg, ycg):
        """
        Compute the thymio angle
        :param xcy:     Yellow square center along X axis in pixel
        :param ycy:     Yellow square center along Y axis in pixel
        :param xcg:     Green square center along X axis in pixel
        :param ycg:     Green square center along Y axis in pixel
        """

        y1 = int(ycy)
        y2 = int(ycg)
        x1 = int(xcy)
        x2 = int(xcg)

        #  Find in which dial thymio is
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

    def detect_and_rotate(self, image):
        """
        Detect the blue square surronding the world in order to turn and resize the picture
        :param image:       Picture to analyse
        """

        # computing of the blue mask to isolate the contours of the map
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask_blue = cv2.inRange(hsv, self.colors.low_blue, self.colors.up_blue)
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
        if maxArea < 10:
            return None

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

        # Correct if needed the angle between vertical and longest size of rectangle
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

        #  Return the corrected grid in an array
        final_grid = np.array(corrected)
        return final_grid

    def test_camera(self):
        # i = 0 pour main webcam aka built in, 1 for first usb port etc
        # cap = cv2.VideoCapture(i)

        cap = cv2.VideoCapture(int(os.getenv("CAMERA_PORT")))

        # cap = cv2.VideoCapture('test.mp4')
        # cap = cv2.VideoCapture('test_video.mov')

        fW = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fH = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Que QR code, a rajouter cadre + depassement
        self.gw = (self.LENGTH + 5)
        self.gh = (self.WIDTH + 5)
        cam_grid_ratio = (self.gw / fW, self.gh / fH)

        while True:

            _, frame = cap.read()
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.colors.low_green, self.colors.up_green)

            # mask = cv2.inRange(hsv, low_green, up_green)
            resg = cv2.bitwise_and(frame, frame, mask)

            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
            areas = [cv2.contourArea(c) for c in contours]
            if len(areas) < 1:

                # Display the resulting frame
                frame = cv2.resize(frame, (0, 0), None, 1, 1)
                cv2.imshow('frame', frame)
                # If "q" is pressed on the keyboard, exit this loop
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            else:

                # Find the largest moving object in the image
                max_index = np.argmax(areas)

                cnt = contours[max_index]
                xg, yg, wg, hg = cv2.boundingRect(cnt)

                # Draw circle in the center of the bounding box

                mask = cv2.inRange(hsv, self.colors.low_yellow, self.colors.up_yellow)
                resy = cv2.bitwise_and(frame, frame, mask)

                contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
                areas = [cv2.contourArea(c) for c in contours]
                if len(areas) < 1:

                    # Display the resulting frame
                    frame = cv2.resize(frame, (0, 0), None, 1, 1)
                    cv2.imshow('frame', frame)
                    masky = cv2.inRange(hsv, self.colors.low_yellow, self.colors.up_yellow)
                    resy = cv2.bitwise_and(frame, frame, masky)
                    maskg = cv2.inRange(hsv, self.colors.low_green, self.colors.up_green)
                    resg = cv2.bitwise_and(frame, frame, maskg)
                    cv2.imshow('resg', maskg)
                    cv2.imshow('resy', masky)

                    # If "q" is pressed on the keyboard, exit this loop
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    max_index = np.argmax(areas)
                    cnt = contours[max_index]
                    xy, yy, wy, hy = cv2.boundingRect(cnt)

                    x2g = xg + int(wg / 2)
                    y2g = yg + int(hg / 2)
                    x2y = xy + int(wy / 2)
                    y2y = yy + int(hy / 2)
                    # cv2.circle(frame,(x2,y2),4,(255,255,0),-1)
                    x2 = int((x2g + x2y) / 2)
                    y2 = int((y2g + y2y) / 2)
                    # print("x2,y2", x2, y2)
                    text = "Robot center in map's squares"
                    cv2.putText(frame, text, (x2 - 120, y2 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    # int(x2*cam_grid_ratio[0])) is the x value in grid coord
                    # self.gh-int(y2*cam_grid_ratio[1]) is the y value in grid coord
                    text2 = "x: " + str(int(x2 * cam_grid_ratio[0])) + ", y: " + str(
                        self.gh - int(y2 * cam_grid_ratio[1]))
                    cv2.putText(frame, text2, (x2 - 50, y2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                    cv2.circle(frame, (x2, y2), 4, (255, 255, 0), -1)

                    # frame = cv2.resize(frame, (0, 0), None, 0.5, 0.5)
                    cv2.imshow('frame', frame)
                    masky = cv2.inRange(hsv, self.colors.low_yellow, self.colors.up_yellow)
                    resy = cv2.bitwise_and(frame, frame, masky)
                    maskg = cv2.inRange(hsv, self.colors.low_green, self.colors.up_green)
                    resg = cv2.bitwise_and(frame, frame, maskg)
                    maskb = cv2.inRange(hsv, self.colors.low_blue, self.colors.up_blue)
                    maskr = cv2.inRange(hsv, self.colors.low_red, self.colors.up_red)
                    cv2.imshow('resg', maskg)
                    cv2.imshow('resy', masky)
                    cv2.imshow('resb', maskb)
                    cv2.imshow('resr', maskr)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cv2.destroyAllWindows()
        cap.release()

    def nothing(self, x):
        pass

    def camera_tweak(self):
        cap = cv2.VideoCapture(int(os.getenv("CAMERA_PORT")))
        cv2.namedWindow("Trackbars")

        cv2.createTrackbar("L – H", "Trackbars", 0, 179, self.nothing)
        cv2.createTrackbar("L – S", "Trackbars", 0, 255, self.nothing)
        cv2.createTrackbar("L – V", "Trackbars", 0, 255, self.nothing)
        cv2.createTrackbar("U – H", "Trackbars", 179, 179, self.nothing)
        cv2.createTrackbar("U – S", "Trackbars", 255, 255, self.nothing)
        cv2.createTrackbar("U – V", "Trackbars", 255, 255, self.nothing)

        while True:
            _, frame = cap.read()
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            l_h = cv2.getTrackbarPos("L – H", "Trackbars")
            l_s = cv2.getTrackbarPos("L – S", "Trackbars")
            l_v = cv2.getTrackbarPos("L – V", "Trackbars")
            u_h = cv2.getTrackbarPos("U – H", "Trackbars")
            u_s = cv2.getTrackbarPos("U – S", "Trackbars")
            u_v = cv2.getTrackbarPos("U – V", "Trackbars")

            lower_blue = np.array([l_h, l_s, l_v])
            upper_blue = np.array([u_h, u_s, u_v])
            mask = cv2.inRange(hsv, lower_blue, upper_blue)

            result = cv2.bitwise_and(frame, frame, mask=mask)

            cv2.imshow("frame", frame)
            cv2.imshow("mask", mask)
            cv2.imshow("result", result)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
