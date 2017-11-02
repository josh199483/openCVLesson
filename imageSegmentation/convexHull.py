import numpy as np
import cv2

# Load image and keep a copy
image = cv2.imread('../images/house.jpg')
orig_image = image.copy()
cv2.imshow('Original Image', orig_image)
cv2.waitKey(0)
# Grayscale and binarize
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#這裡使用cv2.THRESH_BINARY_INV是為了讓圖片變黑底白圖，因為若是白色背景的話，findcontour會把最外層的白色背景當成輪廓之一
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
cv2.imshow('thresh Image', thresh)
# Find contours，使用cv2.RETR_LIST的原因是因為需要看每個輪廓，不只是要看最外層的輪廓
_, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# Iterate through each contour and compute the bounding rectangle
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(orig_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imshow('Bounding Rectangle', orig_image)
cv2.waitKey(0)

# Iterate through each contour and compute the approx contour
for c in contours:
    # 可試著把accuracy的0.03改變大小，越小代表準確度越高，可以畫出更符合形狀的輪廓
    accuracy = 0.03 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, accuracy, True)
    cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)
    cv2.imshow('Approx Poly DP', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
## convex hull代表把圖像中的物體用最外圍的點連接起來
image = cv2.imread('../images/hand.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow('Original Image', image)
cv2.waitKey(0)
# Threshold the image
ret, thresh = cv2.threshold(gray, 176, 255, 0)

# Find contours
__, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
# 因為是白色背景的圖片，所以在findContours的時候，會把背景的方框也當作其中一個輪廓，因此在以面積來排序時把最後一個去掉
n = len(contours) - 1
contours = sorted(contours, key=cv2.contourArea, reverse=False)[:n]

# Iterate through contours and draw the convex hull
for c in contours:
    hull = cv2.convexHull(c)
    cv2.drawContours(image, [hull], 0, (0, 255, 0), 2)
    cv2.imshow('Convex Hull', image)
cv2.waitKey(0)
cv2.destroyAllWindows()