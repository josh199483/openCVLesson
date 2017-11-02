import cv2
import numpy as np

image = cv2.imread('../images/input.jpg')

# 這裡建了一個shape和image一模一樣，但值全為1的陣列，再乘以175等於讓所有陣列值都為175
M = np.ones(image.shape, dtype = "uint8") * 175
print(M)
# 這裡把image和M相加，會呈現一張很接近白色、明亮的圖(接近255)
added = cv2.add(image, M)
cv2.imshow("Added", added)

# 這裡把image和M相減，會呈現一張很接近黑色、暗沉的圖(接近0)
subtracted = cv2.subtract(image, M)
cv2.imshow("Subtracted", subtracted)
cv2.waitKey(0)
cv2.destroyAllWindows()
## bitwise!!!
# Making a sqare
square = np.zeros((300, 300), np.uint8)
cv2.rectangle(square, (50, 50), (250, 250), 255, -2)
cv2.imshow("Square", square)
# Making a ellipse
ellipse = np.zeros((300, 300), np.uint8)
cv2.ellipse(ellipse, (150, 150), (150, 150), 30, 0, 180, 255, -1)
cv2.imshow("Ellipse", ellipse)
cv2.waitKey(0)
cv2.destroyAllWindows()
##切記!!!做bitwise一定要是相同大小的圖片
# 呈現and的部分(呈現白色的部分因為值是255)
bitwiseAnd = cv2.bitwise_and(square, ellipse)
cv2.imshow("AND", bitwiseAnd)
cv2.waitKey(0)

# 呈現or的部分
bitwiseOr = cv2.bitwise_or(square, ellipse)
cv2.imshow("OR", bitwiseOr)
cv2.waitKey(0)

# 呈現not or的部分
bitwiseXor = cv2.bitwise_xor(square, ellipse)
cv2.imshow("XOR", bitwiseXor)
cv2.waitKey(0)

# 呈現not的部分，參數只需帶一個image，因為是呈現相反的
bitwiseNot_sq = cv2.bitwise_not(square)
cv2.imshow("NOT - square", bitwiseNot_sq)
cv2.waitKey(0)

### Notice the last operation inverts the image totally
cv2.destroyAllWindows()