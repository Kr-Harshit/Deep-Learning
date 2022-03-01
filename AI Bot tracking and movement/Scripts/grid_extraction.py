import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import tensorflow as tf
from tqdm import tqdm
 

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


def stackImages(imgArray,scale):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        hor_con= np.concatenate(imgArray)
        ver = hor
    return ver

def get_grid(cells):
    cellClassifier = tf.keras.models.load_model('models/cellClassiffer.h5')
    labels = {0:2, 1:1, 2:-1, 3:0}
    grid = []
    for i in range(len(cells)):
        row_grid = []
        for j in range(len(cells[i])):
            shape = (1, cells[i][j].shape[0], cells[i][j].shape[1], cells[i][j].shape[2])
            x = cellClassifier.predict(cells[i][j].reshape(shape))[0].argmax()
            print(i, j, '-->', labels[x])
            row_grid.append(int(labels[x]))
        grid.append(row_grid)

    return np.asarray(grid,  dtype=np.int64)

def make_cell_dataset():
    IMG_HEIGHT = 1000
    IMG_WIDTH = 1000
    IMG_PATH  = './Datasets/grid_images'
    dataset_path = './Datasets/cell_dataset'

    
    for img in tqdm(os.listdir(IMG_PATH)):
        img_file = os.path.join(IMG_PATH, img)
        gr = GridReader(img_file , IMG_HEIGHT, IMG_WIDTH)
        mask = gr.mask_grid(gr.change_to_bird_view())
        cells = gr.get_cells(mask)

        for i in range(len(cells)):
            for j in range(len(cells[i])):
                if i == 0 and (j in range(7, 11)):
                    label = "end"
                elif i in range(0,8) and (j in range(0, 7) or j in range(11, 18)):
                    label = "not walkable"
                else :
                    label = "walkable"
                filename = f'[{i},{j}]_{img}'
                if not os.path.exists(os.path.join(dataset_path, label)):
                    os.makedirs(os.path.join(dataset_path, label)) 
                cv2.imwrite(os.path.join(dataset_path, label, filename), cells[i][j])
    
    print("*********Completed******")

def make_grid_mask_dataset():
    IMG_HEIGHT = 1000
    IMG_WIDTH = 1000
    IMG_PATH  = './Datasets/test/raw'
    dataset_path = './Datasets/test/masks'

    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    for img in tqdm(os.listdir(IMG_PATH)):
        img_file = os.path.join(IMG_PATH, img)
        gr = GridReader(img_file , IMG_HEIGHT, IMG_WIDTH)
        mask = gr.mask_grid(gr.change_to_bird_view())
        cv2.imwrite(os.path.join(dataset_path, img), mask)


def main():
    IMG_HEIGHT = 1000
    IMG_WIDTH = 1000

    img_path = 'pics'
    img_file = '5.jpeg'

    gr = GridReader(os.path.join(img_path, img_file) , IMG_HEIGHT, IMG_WIDTH)
    # img_thres = gr.preProcess_image()

    # contours, hierarchy = cv2.findContours(img_thres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # img_countor = img.copy()
    # cv2.drawContours(img_countor, contours, -1, (0, 0, 255), 5)

    # corners, biggest = gr.get_biggest_contour(img_thres)
    # img_big_contours = img.copy()
    # cv2.drawContours(img_big_contours, corners, -1, (0, 0, 255), 5)
    # # cv2.drawContours(img_big_contours, biggest, -1, (0, 255, 0), 2)

    # imageArray = ([img, img_thres, img_countor, img_big_contours])
    # stackedImage = stackImages(imageArray, 0.6)
    # cv2.imshow('Stacked Images', stackedImage)
    # cv2.waitKey(0)

    img_wrapped = gr.change_to_bird_view()
    mask = gr.mask_grid(img_wrapped)

    # -----getting image of each cells
    cells = gr.get_cells(mask)
   
    # print("*")
    # grid = get_grid(cells)
    # np.savetxt('grid.txt', grid, fmt="%d")
    # pprint(grid)

    # _, axes = plt.subplots(10, 18, sharex=True, sharey=True)
    # for i in range(10):
    #     for j in range(18):
    #         axes[i, j].imshow(cells[i][j])
    #         axes[i, j].axis("off")

    # plt.show()


if __name__ == "__main__":
    make_grid_mask_dataset()

