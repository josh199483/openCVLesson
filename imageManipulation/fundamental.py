import cv2
import numpy as np

image = cv2.imread('../images/input.jpg')
print(image.shape)
print("BGR height pixels is:",image.shape[0])
print("BGR width pixels is:",image.shape[1])
cv2.imshow("BGR image",image)
gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow("gray image",gray_image)
# color space,印出pixel在(x,y)為(0,0)的BGR值
B,G,R = image[0,0]
print(B,G,R)
i = image[0]
print(i)
# gray_image 像素只有一個值
print(gray_image[0,0])
# change to HSV image
hsv_image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
cv2.imshow("hsv image",hsv_image)
cv2.imshow("hue image",hsv_image[:,:,0]) #hue channel
cv2.imshow("saturation image",hsv_image[:,:,1]) #saturation channel，形狀看得還算清楚
cv2.imshow("value image",hsv_image[:,:,2]) #value channel，這張圖比較清楚，還看的出來原型
##split 使用
Bimage, Gimage, Rimage = cv2.split(image)
print (Bimage.shape)
cv2.imshow("Blue", Bimage)  #三種顏色切分開來的圖看不太出差異，因為每張圖像素值都變成一個值，等於做了灰階轉換!!
cv2.imshow("Green", Gimage)
cv2.imshow("Red", Rimage)
##merge 使用
# Let's re-make the original image
merged = cv2.merge([Bimage, Gimage, Rimage])
cv2.imshow("Merged", merged)

# Let's amplify the blue color，增強藍色像素值
blue_merged = cv2.merge([Bimage+100, Gimage, Rimage])
cv2.imshow("Merged with Blue Amplified", blue_merged) #可明顯看出是藍色的圖像
##建立BGR的圖像
#此處用numpy建立一個二維陣列，值全為0，shape為前幾張圖的長跟寬，不取第三維的值(也就是BGR)
zeros = np.zeros(image.shape[:2], dtype = "uint8")
#此處的幾張圖只會出現對應的顏色
cv2.imshow("all Blue", cv2.merge([Bimage, zeros, zeros]))
cv2.imshow("all Green", cv2.merge([zeros, Gimage, zeros]))
cv2.imshow("all Red", cv2.merge([zeros, zeros, Rimage]))

cv2.waitKey(0)
cv2.destroyAllWindows()