import cv2
import numpy as np

# Let's load a simple image with 3 black squares
image = cv2.imread('../images/shapes.jpg')
cv2.imshow('Input Image', image)
cv2.waitKey(0)

# Grayscale
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# Find Canny edges
edged = cv2.Canny(gray, 30, 200)
cv2.imshow('Canny Edges', edged)
cv2.waitKey(0)

# Finding Contours
# Use a copy of your image e.g. edged.copy(), since findContours alters the image
#此處opencv3有改寫一點語法，回傳的值變3個
#第二個參數是定義hierarchy的，最常用的是external(範例)，或是list，可以決定該圖像的contour是要全部顯示還是只顯示最外層的
#，從最下面的例子就可知道是甚麼意思了
#第三個參數有兩種選擇(Approximation Methods)，另外一種是cv2.CHAIN_APPROX_SIMPLE，第一種會把contour的每個點都找出來，
# 第二種則是把contour的交點位置回傳(三角形有三個點)，選擇simple運算速度較快，但複雜圖形可能無法較好的取得contour
#回傳的hierarchy紀錄的是contour和contour之間(同一個contour 點之間的關係)的關係
image_con, contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# (_, contours, _) = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.imshow('Canny Edges After Contouring', edged)
cv2.waitKey(0)
print(contours)
#可以看到contour是一個三維陣列，最外層那一維代表有幾個圖像的contour，之後代表的是x,y像素位置
print("Number of Contours found = " + str(len(contours)))
# Draw all contours，第三個參數選擇-1把所有contour都標示出來
cv2.drawContours(image, contours, -1, (0,255,0), 3)
cv2.imshow('Contours', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#灰階
image = cv2.imread('../images/shapes_donut.jpg',1)
cv2.imshow('Input Image', image)
cv2.waitKey(0)
edged = cv2.Canny(image, 30, 200)
#第二參數做點改變，才能找到內部的contour
(_, contours, _) = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(image, contours, -1, (0,255,0), 3)
cv2.imshow('Contours', image)
cv2.waitKey(0)
cv2.destroyAllWindows()