import cv2
import numpy as np

##template matching事實上並不是實用的方法，因為有時候實際的圖是會scale、rotation、brightness、affine等等，只要有一些差異就會比對不到
# Load input image and convert to grayscale
image = cv2.imread('../images/WaldoBeach.jpg')
cv2.imshow('Where is Waldo?', image)
cv2.waitKey(0)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Load Template image
template = cv2.imread('../images/waldo.jpg',0)
cv2.imshow('template', template)
cv2.waitKey(0)
##用灰階圖來做比對，這裡選擇correlation and coefficient方法
result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF)
#此方法也是類似回傳找到物件的左上座標
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
print(min_loc)
print(max_loc)
#畫個方框作標示
top_left = max_loc
bottom_right = (top_left[0] + 50, top_left[1] + 50)
cv2.rectangle(image, top_left, bottom_right, (0,0,255), 5)

cv2.imshow('Where is Waldo?', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
##http://docs.opencv.org/2.4/modules/imgproc/doc/object_detection.html