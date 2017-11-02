# import numpy as np
# import cv2
#
# camimage = cv2.VideoCapture(0)
# # 確認鏡頭有開啟
# while (camimage.isOpened()):
#     ret, img = camimage.read()
#     if ret == True:
#         cv2.imshow("test", img)
#         # 每0.1秒檢查一次有沒有按
#         k = cv2.waitKey(100)
#         if k == ord("z") or k == ord("Z"):
#             cv2.imwrite("test.jpg", img)

# 裝飾器撰寫
def print_fun(title):
	def decorator(func):
		def modified_func(*args, **kwargs):
			result = func(*args, ** kwargs)
			print(title, result)
		return modified_func
	return decorator
@print_fun(title='title:')
def add(*tup):
	return sum(tup)
# 以下兩者相同
add(1, 2, 3, 4)


