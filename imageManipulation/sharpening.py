import cv2
import numpy as np

##sharpening(銳化)，sharpening可使圖片的邊緣變得更清晰
image = cv2.imread('../images/input.jpg')
cv2.imshow('Original', image)

# Create our shapening kernel, we don't normalize since the
# the values in the matrix sum to 1
#這邊還不太清楚為何銳化要使用這樣的矩陣
kernel_sharpening = np.array([[-1,-1,-1],
                              [-1,9,-1],
                              [-1,-1,-1]])

# applying different kernels to the input image
sharpened = cv2.filter2D(image, -1, kernel_sharpening)

cv2.imshow('Image Sharpening', sharpened)

cv2.waitKey(0)
cv2.destroyAllWindows()
##這邊是使用Dilation, Erosion, Opening and Closing
image = cv2.imread('../images/opencv_inv.png', 0)
cv2.imshow('Original', image)
cv2.waitKey(0)

# Let's define our kernel size
kernel = np.ones((5,5), np.uint8)

# Now we erode
erosion = cv2.erode(image, kernel, iterations = 1)
cv2.imshow('Erosion', erosion)
cv2.waitKey(0)
#dilate
dilation = cv2.dilate(image, kernel, iterations = 1)
cv2.imshow('Dilation', dilation)
cv2.waitKey(0)

# Opening - Good for removing noise，erosion then dilation
opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
cv2.imshow('Opening', opening)
cv2.waitKey(0)

# Closing - Good for removing noise，dilation then erosion，會和original大致上差不多
closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
cv2.imshow('Closing', closing)
cv2.waitKey(0)
cv2.destroyAllWindows()
# There are some other less popular morphology operations, see the official OpenCV site:
# http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html