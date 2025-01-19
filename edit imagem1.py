import cv2
import numpy as np

image = cv2.imread("rinoceronte.jpg")

kernel = np.ones((5, 5), np.uint8)  # 5x5

small_kernel = np.ones((3, 3), np.uint8)  # 3x3

hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

h, s, v = cv2.split(hsv_image)

sat_mask = cv2.inRange(s, 0, 100)

value_mask = cv2.inRange(v, 70, 255)

mask = cv2.bitwise_and(sat_mask, value_mask)

eroded_mask = cv2.erode(mask, kernel, iterations=15)

opened_mask = cv2.dilate(eroded_mask, kernel, iterations=15)

better_mask = cv2.bitwise_and(cv2.dilate(opened_mask, kernel, iterations=1), sat_mask)

for i in range(350):
    better_mask = cv2.bitwise_and(cv2.dilate(better_mask, small_kernel, iterations=1), sat_mask)

cv2.imshow("Mask", better_mask)

cv2.imwrite("rinoceronte_segmentado.jpg", better_mask)
