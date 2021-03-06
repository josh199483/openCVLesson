import cv2
import numpy as np
import matplotlib.pyplot as plt
##Getting Perpsective Transform
image = cv2.imread('../images/scan.jpg')
cv2.imshow('Original', image)
cv2.waitKey(0)
#可看出這張圖裡的紙，並沒有對邊平行，所以這邊要做個轉換
# Cordinates of the 4 corners of the original image
points_A = np.float32([[320, 15], [700, 215], [85, 610], [530, 780]])

# Cordinates of the 4 corners of the desired output
# We use a ratio of an A4 Paper 1 : 1.41
points_B = np.float32([[0, 0], [420, 0], [0, 594], [420, 594]])

# Use the two sets of four points to compute
# the Perspective Transformation matrix, M
M = cv2.getPerspectiveTransform(points_A, points_B)
warped = cv2.warpPerspective(image, M, (420, 594))
cv2.imshow('warpPerspective', warped)
cv2.waitKey(0)
cv2.destroyAllWindows()
##In affine transforms you only need 3 coordiantes to obtain the correct transform
image = cv2.imread('../images/ex2.jpg')
rows, cols, ch = image.shape
cv2.imshow('Original', image)
cv2.waitKey(0)
# Cordinates of the 4 corners of the original image
points_A = np.float32([[320, 15], [700, 215], [85, 610]])

# Cordinates of the 4 corners of the desired output
# We use a ratio of an A4 Paper 1 : 1.41
points_B = np.float32([[0, 0], [420, 0], [0, 594]])

# Use the two sets of four points to compute
# the Perspective Transformation matrix, M
M = cv2.getAffineTransform(points_A, points_B)
warped = cv2.warpAffine(image, M, (cols, rows))
#這個畫出來的圖好像不是平行的?
cv2.imshow('warpPerspective', warped)
cv2.waitKey(0)
cv2.destroyAllWindows()