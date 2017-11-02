## Blob Detection
import cv2
import numpy as np
##blob是甚麼呢?在圖片裡會有一群共享某特徵的像素群，就稱為blob
##流程是1.先創造一個blob detector 2.再把圖片放進detector 3.得到關鍵特徵 4.把特徵的點畫出來
# Read image
image = cv2.imread("../images/Sunflowers.jpg",0)

# Set up the detector with default parameters，方法名稱有改過
detector = cv2.SimpleBlobDetector_create()

# Detect blobs.
keypoints = detector.detect(image)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of
# the circle corresponds to the size of blob
blank = np.zeros((1, 1))
# blobs = cv2.drawKeypoints(image, keypoints, blank, (0, 255, 255),cv2.DRAW_MATCHES_FLAGS_DEFAULT)
blobs = cv2.drawKeypoints(image, keypoints, blank, (0, 255, 255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show keypoints
cv2.imshow("Blobs", blobs)
cv2.waitKey(0)
cv2.destroyAllWindows()