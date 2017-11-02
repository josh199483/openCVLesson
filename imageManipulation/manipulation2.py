import cv2
import numpy as np

##試想當想要把圖片從小變大，總不可能是像素變大，那縮放這種效果是要如何達成就要使用interpolation，
##類似差補法的方式，讓圖片能用某種計算方式來填補放大後空缺的像素值
image = cv2.imread('../images/input.jpg')
# Let's make our image 3/4 of it's original size，等比例縮放
image_scaled = cv2.resize(image, None, fx=0.75, fy=0.75) #default=cv2.INTER_LINEAR,cv2.INTER_NEAREST速度最快
cv2.imshow('Scaling - Linear Interpolation', image_scaled)

# Let's double the size of our image
img_scaled = cv2.resize(image, None, fx=2, fy=2, interpolation = cv2.INTER_CUBIC ) #better,cv2.INTER_LANCZOS4效果最好
cv2.imshow('Scaling - Cubic Interpolation', img_scaled)

# Let's skew the re-sizing by setting exact dimensions
img_scaled = cv2.resize(image, (900, 400), interpolation = cv2.INTER_AREA)
cv2.imshow('Scaling - Skewed Size', img_scaled)
cv2.waitKey()
cv2.destroyAllWindows()
## image pyramids
smaller = cv2.pyrDown(image)
larger = cv2.pyrUp(smaller)
#每次縮小放大，長寬都差距兩倍
cv2.imshow('Original', image )

cv2.imshow('Smaller ', smaller )
cv2.imshow('Larger ', larger ) #用這種方法看到的larger圖片會比原圖來的模糊!!
cv2.waitKey()
cv2.destroyAllWindows()
## image cropping(圖片裁切)，opencv內建是沒有crop方法的，此處使用numpy取值的方式，也可使用PIL套件的Image的crop()
height, width = image.shape[:2]
# Let's get the starting pixel coordiantes (top  left of cropping rectangle)
start_row, start_col = int(height * .25), int(width * .25)
# Let's get the ending pixel coordinates (bottom right)
end_row, end_col = int(height * .75), int(width * .75)
# Simply use indexing to crop out the rectangle we desire
cropped = image[start_row:end_row , start_col:end_col]

cv2.imshow("Original Image", image)
cv2.imshow("Cropped Image", cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()