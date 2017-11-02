import cv2
import numpy as np

##Thresholding, Binarization & Adaptive Thresholding
#cv2.threshold(image,threshold value,max value,threshold type)
# Load our image as greyscale(以灰階模式開啟)
image = cv2.imread('../images/gradient.jpg', 0)
cv2.imshow('Original', image)

# Values below 127 goes to 0 (black, everything above goes to 255 (white)(這就是作二值化，只有黑白兩種)
ret, thresh1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
cv2.imshow('1 Threshold Binary', thresh1)

# Values below 127 go to 255 and values above 127 go to 0 (reverse of above)(與上面一種相反)
ret, thresh2 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
cv2.imshow('2 Threshold Binary Inverse', thresh2)

# Values above 127 are truncated (held) at 127 (the 255 argument is unused)(超過127的都變成127，255是不必要的參數)
ret, thresh3 = cv2.threshold(image, 127, 255, cv2.THRESH_TRUNC)
cv2.imshow('3 THRESH TRUNC', thresh3)

# Values below 127 go to 0, above 127 are unchanged(低於127的都變為0，其餘照舊，255一樣是無用)
ret, thresh4 = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO)
cv2.imshow('4 THRESH TOZERO', thresh4)

# Resever of above, below 127 is unchanged, above 127 goes to 0(與上一種相反，超過127的都變為0，其餘照舊)
ret, thresh5 = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO_INV)
cv2.imshow('5 THRESH TOZERO INV', thresh5)
cv2.waitKey(0)
cv2.destroyAllWindows()

##adaptive threshold比較
# Load our new image
image = cv2.imread('../images/Origin_of_Species.jpg', 0)

cv2.imshow('Original', image)
cv2.waitKey(0)

# Values below 127 goes to 0 (black, everything above goes to 255 (white)
ret,thresh1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
cv2.imshow('Threshold Binary', thresh1)
cv2.waitKey(0)

# It's good practice to blur images as it removes noise
image = cv2.GaussianBlur(image, (3, 3), 0)

# Using adaptiveThreshold(以下三種方式，實際應用上都可以試試看比較效果再決定)
#cv2.adaptiveThreshold(image,Max Value,Adaptive Type,Threshold Type,Block Size,constant that is substracted from mean通常都用5)
thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY, 3, 5)
cv2.imshow("Adaptive Mean Thresholding", thresh)
cv2.waitKey(0)

_, th2 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow("Otsu's Thresholding", thresh)
cv2.waitKey(0)

# Otsu's thresholding after Gaussian filtering，這個效果都常最好，因為會先用直方圖，去判斷兩個最常出現的像素值區間取最容易切分的值
#當作threshold，但實際應用上就要看當時情景來作取捨，因有時不需要這麼精準可以選擇運算速度快的方法

blur = cv2.GaussianBlur(image, (5,5), 0)
_, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow("Guassian Otsu's Thresholding", thresh)
cv2.waitKey(0)

cv2.destroyAllWindows()