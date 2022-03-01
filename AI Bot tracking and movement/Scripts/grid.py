import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class GridReader:
    def __init__(self, img_path, img_height, img_width):
        self.img_path = img_path
        self.IMG_HEIGHT = img_height
        self.IMG_WIDTH = img_width

        self.block_width = int(self.IMG_WIDTH /18)
        self.grid_width = 18 * self.block_width  # 18 -> max blocks in width of grid
        self.grid_height = 10 * self.block_width # 10 ->  max blocks in height of grid
     
        self.img = self.__read_image()

    def __read_image(self):
        img = cv2.imread(self.img_path)
        img = cv2.resize(img, (self.IMG_WIDTH, self.IMG_HEIGHT))
        return img

    def preProcess_image(self):
        img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        # img_blur = cv2.bilateralFilter(img_gray,  5, 75, 75)
        img_blur = cv2.GaussianBlur(img_gray, (7, 7), 1)
        img_threshold = cv2.adaptiveThreshold(img_blur, 255, 1, 1, 5, 3)
        return img_threshold    

    def gird_corner(self):
        pass

    def get_biggest_contour(self, img_thres):
        contours, hierarchy = cv2.findContours(img_thres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = 0
        corners, biggest = None, None
        for i in contours:
            area = cv2.contourArea(i)
            if area >= 60 * (self.block_width**2):
                peri = cv2.arcLength(i, True)  #perimeter
                approx = cv2.approxPolyDP(i, 0.02 * peri, True)
                if area > max_area :
                    corners = approx
                    max_area = area
                    biggest = i
        return corners, biggest

    def __reorder_points(self, points):
        points = points.reshape((-1, 2))
        points_new = np.zeros((4, 1, 2), dtype=np.int32)
        add = points.sum(1)
        points_new[0] = points[np.argmin(add)]
        points_new[3] = points[np.argmax(add)]
        diff = np.diff(points, axis=1)
        points_new[1] = points[np.argmin(diff)]
        points_new[2] = points[np.argmax(diff)]
        return points_new

    def __wrap_perspective(self, img_thres):
        corners, biggest = self.get_biggest_contour(img_thres)
        corners = self.__reorder_points(corners)
        pts1 = np.float32(corners)
        pts2 = np.float32([
            [self.block_width*7, 0],
            [self.block_width*11, 0],
            [0, self.grid_height],
            [self.grid_width, self.grid_height]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        img_wrapped = cv2.warpPerspective(self.img, matrix, (self.grid_width, self.grid_height))
        return img_wrapped

    def get_image(self):
        return self.img

    def change_to_bird_view(self):
        img_thres = self.preProcess_image()
        img_wrapped = self.__wrap_perspective(img_thres)
        return img_wrapped

    def __mask_grid(self, img):
        mask = img.copy()
        cells = []

        mask[0 : 8*self.block_width, 0: 7*self.block_width, :] = 0
        mask[0 : 8*self.block_width, 11*self.block_width : self.grid_width, :] = 0

        return mask

    def get_cells(self, img):
        cells = []

        for i in range(0, self.grid_height, self.block_width):
            cols = []
            for j in range(0, self.grid_width, self.block_width):
                cols.append(img[i:i+self.block_width, j:j+ self.block_width, :])
            cells.append(cols)
        return cells


