##Sorting Contours
import cv2
import numpy as np

# Load our image
image = cv2.imread('../images/bunchofshapes.jpg')
cv2.imshow('0 - Original Image', image)
cv2.waitKey(0)

# Create a black image with same dimensions as our loaded image
blank_image = np.zeros((image.shape[0], image.shape[1], 3))
# Create a copy of our original image
orginal_image = image
# Grayscale our image
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# Find Canny edges
edged = cv2.Canny(gray, 50, 200)
cv2.imshow('1 - Canny Edges', edged)
cv2.waitKey(0)

# Find contours and print how many were found
_, contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#總共找到幾個contour
print ("Number of contours found = ", len(contours))

#Draw all contours
cv2.drawContours(blank_image, contours, -1, (0,255,0), 3)
cv2.imshow('2 - All Contours over blank image', blank_image)
cv2.waitKey(0)

# Draw all contours over blank image
cv2.drawContours(image, contours, -1, (0,255,0), 3)
cv2.imshow('3 - All Contours', image)
cv2.waitKey(0)

cv2.destroyAllWindows()
##利用opencv的contourArea方法，計算出物體的面積，藉由找出一些圖像的特徵(面積、周長、質心等)，進行後續處理或機器學習應用
##Function we'll use to display contour area
#111111111111111111111111111111111111
def get_contour_areas(contours):
    # returns the areas of all contours as list
    all_areas = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        all_areas.append(area)
    return all_areas
#自己寫一個計算周長的function，2222222222222222222222222
def get_contour_arclength(contours):
    all_arcs = []
    for cnt in contours:
        arc = cv2.arcLength(cnt,True)
        all_arcs.append(arc)
    return all_arcs
# Let's print the areas of the contours before sorting
print ("Contor Areas before sorting", get_contour_areas(contours))
print(len(contours[0]),len(contours[1]),len(contours[2]))
#印出各輪廓的周長
print("all arcs",get_contour_arclength(contours))
# Sort contours large to small
sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
print ("Contor Areas after sorting", get_contour_areas(sorted_contours))

# Iterate over our contours and draw one at a time
for c in sorted_contours:
    cv2.drawContours(orginal_image, [c], -1, (255,0,0), 3)
    #依照面積大到小的圖形，用藍色來畫出輪廓
    cv2.imshow('Contours by area', orginal_image)
    cv2.waitKey(0)
cv2.destroyAllWindows()

##Functions we'll use for sorting by position
def x_cord_contour(contour):
    # 回傳該輪廓的x座標，此方法的參數寫為contour以作區分，因為此方法是傳進單一個contour為二維陣列(不是contours的三維陣列)
    if cv2.contourArea(contour) > 10:
        M = cv2.moments(contour)
        return (int(M['m10'] / M['m00']))

#3333333333333333333333333333333
def label_contour_center(image, c):
    # Places a red circle on the centers of contours
    #計算質心位置，參數是給一個二維陣列(這裡是x,y座標)
    M = cv2.moments(c)
    cx = int(M['m10'] / M['m00']) #x座標
    cy = int(M['m01'] / M['m00']) #y座標

    # Draw the countour number on the image
    cv2.circle(image, (cx, cy), 10, (0, 0, 255), -1)
    return image
# print(x_cord_contour(contours))

# 按照質心的位置把每個輪廓都畫上質心
for (i, c) in enumerate(contours):
    orig = label_contour_center(image, c)

cv2.imshow("4 - Contour Centers ", image)
cv2.waitKey(0)

# 以質心x座標小到大排序，因opencv座標是由左上角開始算起
contours_left_to_right = sorted(contours, key=x_cord_contour, reverse=False)

#從左到右標記contour的編號
for (i, c) in enumerate(contours_left_to_right):
    cv2.drawContours(orginal_image, [c], -1, (0, 0, 255), 3)
    M = cv2.moments(c)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    cv2.putText(orginal_image, str(i + 1), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('6 - Left to Right Contour', orginal_image)
    cv2.waitKey(0)
    #為了儲存圖片，把每個輪廓的都裁切成方形
    (x, y, w, h) = cv2.boundingRect(c)

    # Let's now crop each contour and save these images
    cropped_contour = orginal_image[y:y + h, x:x + w]
    image_name = "output_shape_number_" + str(i + 1) + ".jpg"
    print(image_name)
    cv2.imwrite(image_name, cropped_contour)

cv2.destroyAllWindows()