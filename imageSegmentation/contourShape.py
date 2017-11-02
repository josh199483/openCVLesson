import cv2
import numpy as np

## Shape Matching
# cv2.matchShapes(contour template, contour, method, method parameter)
# Return Value – match value (lower values means a closer match)
# Contour Template – This is our reference contour that we’re trying to find in the new image
# Contour – The individual contour we are checking against
# Method – Type of contour matching (1, 2, 3)
# Method Parameter – leave alone as 0.0 (not fully utilized in python OpenCV)

# Load the shape template or reference image
template = cv2.imread('../images/4star.jpg', 0)
cv2.imshow('Template', template)
cv2.waitKey()
# Load the target image with the shapes we're trying to match
target = cv2.imread('../images/shapestomatch.jpg')
target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

# Threshold both images first before using cv2.findContours
ret, thresh1 = cv2.threshold(template, 127, 255, 0)
ret, thresh2 = cv2.threshold(target_gray, 127, 255, 0)

# 先找出template的contour，並且依照面積大小排序
_, contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
# 只取第二個，因為不需要白色背景的輪廓
template_contour = contours[1]

# Extract contours from second target image
_, contours, hierarchy = cv2.findContours(thresh2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
    # template的contour依序跟target的contour比較相似度，值越小越相似
    # 第3個參數是選擇不同方法，可試試看
    match = cv2.matchShapes(template_contour, c, 3, 0.0)
    print(match)
    # 這裡雖然取小於0.15
    if match < 0.15:
        closest_contour = c
    else:
        closest_contour = []

cv2.drawContours(target, [closest_contour], -1, (0, 255, 0), 3)
cv2.imshow('Output', target)
cv2.waitKey()
cv2.destroyAllWindows()
##此網站有更詳細資訊，http://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html