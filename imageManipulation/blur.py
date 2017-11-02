import cv2
import numpy as np

##Convolutions and Blurring 基礎用法
image = cv2.imread('../images/elephant.jpg')
cv2.imshow('Original Image', image)
cv2.waitKey(0)
#可以看到kernel size選的越大，模糊化程度越高
# Creating our 3 x 3 kernel
kernel_3x3 = np.ones((3, 3), np.float32) / 9

# We use the cv2.fitler2D to conovlve the kernal with an image
blurred = cv2.filter2D(image, -1, kernel_3x3)
cv2.imshow('3x3 Kernel Blurring', blurred)
cv2.waitKey(0)

# Creating our 7 x 7 kernel
kernel_7x7 = np.ones((7, 7), np.float32) / 49

blurred2 = cv2.filter2D(image, -1, kernel_7x7)
cv2.imshow('7x7 Kernel Blurring', blurred2)
cv2.waitKey(0)
cv2.destroyAllWindows()
##使用不同的模糊化方法，各有各的優缺點
#把kernel size裡的值全部加總作標準化
blur = cv2.blur(image, (3,3))
cv2.imshow('Averaging', blur)
cv2.waitKey(0)

# Instead of box filter, gaussian kernel
#高斯模糊雖然kernel size選的大，但模糊的結果跟一般blur差不多
Gaussian = cv2.GaussianBlur(image, (7,7), 0)
cv2.imshow('Gaussian Blurring', Gaussian)
cv2.waitKey(0)

# Takes median of all the pixels under kernel area and central
# element is replaced with this median value
median = cv2.medianBlur(image, 5)
cv2.imshow('Median Blurring', median)
cv2.waitKey(0)

# Bilateral is very effective in noise removal while keeping edges sharp
#這個模糊方式可以很好的去除噪音，畫面上可以看到線條的部分變得更明顯，而耳朵的部分較模糊
bilateral = cv2.bilateralFilter(image, 9, 75, 75)
cv2.imshow('Bilateral Blurring', bilateral)
cv2.waitKey(0)
cv2.destroyAllWindows()

##Image De-noising - Non-Local Means Denoising，去除許多noise
# Parameters, after None are - the filter strength 'h' (5-10 is a good range)
# Next is hForColorComponents, set as same value as h again
dst = cv2.fastNlMeansDenoisingColored(image, None, 6, 6, 7, 21)

cv2.imshow('Fast Means Denoising', dst)
cv2.waitKey(0)

cv2.destroyAllWindows()
# cv2.fastNlMeansDenoising() - works with a single "grayscale" images
# cv2.fastNlMeansDenoisingColored() - works with a "color" image.
# cv2.fastNlMeansDenoisingMulti() - works with "image sequence captured in short period" of time (grayscale images)
# cv2.fastNlMeansDenoisingColoredMulti() - same as above, but for color images.