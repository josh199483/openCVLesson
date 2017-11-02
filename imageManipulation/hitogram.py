import cv2
import numpy as np
from matplotlib import pyplot as plt

image1 = cv2.imread('../images/input.jpg')
#這邊還需了解各參數意義!!!
histogram1 = cv2.calcHist([image1], [0], None, [256], [0, 256])
image2 = cv2.imread('../images/tobago.jpg')
histogram2 = cv2.calcHist([image2], [0], None, [256], [0, 256])
cv2.imshow("input", image1)
cv2.imshow("Tobago", image2)

# We plot a histogram, ravel() flatens our image array
#此處為了畫長條圖需把image的三維矩陣轉為一維矩陣(攤平)，可看出BGR所有的直方圖
print(image1.ravel())
plt.hist(image1.ravel(), 256, [0, 256])
plt.show()

# 之後要以各顏色(BGR)來區分直方圖
color = ('b', 'g', 'r')

#畫各個顏色的直方圖分布
for i, col in enumerate(color):
    histogram_each = cv2.calcHist([image1], [i], None, [256], [0, 256])
    plt.plot(histogram_each, color=col)
    plt.xlim([0, 256])
#此圖可看出紅色是high intensity
plt.show()

#跟另一張圖做比較，可發現顏色對比很明顯
for i, col in enumerate(color):
    histogram_each = cv2.calcHist([image2], [i], None, [256], [0, 256])
    plt.plot(histogram_each, color=col)
    plt.xlim([0, 256])
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()


