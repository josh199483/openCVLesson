import cv2
import numpy as np

##這個還不是很懂....，畫出來的圖怪怪的....
image = cv2.imread('../images/bottlecaps.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blur = cv2.medianBlur(gray, 5)
circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1.5, 10)
circles = np.uint16(np.around(circles))
for i in circles[0, :]:
    # draw the outer circle
    cv2.circle(image, (i[0], i[1]), i[2], (255, 0, 0), 2)

    # draw the center of the circle
    cv2.circle(image, (i[0], i[1]), 2, (0, 255, 0), 5)

cv2.imshow('detected circles', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

##https://docs.opencv.org/3.0-beta/modules/imgproc/doc/feature_detection.html?highlight=houghcircle#cv2.HoughCircles