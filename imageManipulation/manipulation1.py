import cv2
import numpy as np

image = cv2.imread('../images/input.jpg')
## 1. here display translation(圖像位移)!!!!
# Store height and width of the image
height, width = image.shape[:2]

quarter_height, quarter_width = height/4, width/4

#  T  =  | 1 0 Tx |
#        | 0 1 Ty |  是一個二維陣列
# 寬度位移Tx距離，高度位移Ty距離
# T is our translation matrix
T = np.float32([[1, 0, quarter_width], [0, 1,quarter_height]])
print(T)
# We use warpAffine to transform the image using the matrix, T
img_translation = cv2.warpAffine(image, T, (width, height))
cv2.imshow('Translation', img_translation)
## 2.Rotations , cv2.getRotationMatrix2D(rotation_center_x, rotation_center_y, angle of rotation, scale) anticlockwise

# Divide by two to rototate the image around its centre
rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), 90, 0.5) #最後的參數可同時做縮放，1為原始大小
#若縮放調整為0.5，而圖上不想要有黑邊，需直接調整大小
rotated_image1 = cv2.warpAffine(image, rotation_matrix, (width, height))
cv2.imshow('Rotated Image', rotated_image1)
# another method
rotated_image2 = cv2.transpose(image)
#大小一模一樣
cv2.imshow('Rotated Image - Method 2', rotated_image2)
#左右翻轉
flipped = cv2.flip(image, 1)
cv2.imshow('Horizontal Flip', flipped)

cv2.waitKey()
cv2.destroyAllWindows()