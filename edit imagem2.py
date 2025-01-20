import cv2
import numpy as np

image = cv2.imread("zebra.jpg")

kernel = np.ones((5, 5), np.uint8)  # 5x5

small_kernel = np.ones((3, 3), np.uint8)  # 3x3

hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

h, s, v = cv2.split(hsv_image)

sat_mask = cv2.inRange(s, 0, 100)

value_mask = cv2.inRange(v, 70, 255)

mask = cv2.bitwise_and(sat_mask, value_mask)

eroded_mask = cv2.erode(mask, kernel, iterations=5)

opened_mask = cv2.dilate(eroded_mask, kernel, iterations=5)

better_mask = cv2.bitwise_and(cv2.dilate(opened_mask, kernel, iterations=1), sat_mask)

for i in range(350):
    better_mask = cv2.bitwise_and(cv2.dilate(better_mask, small_kernel, iterations=1), sat_mask)


strict_value_mask = cv2.inRange(v, 157, 255)

best_mask = cv2.bitwise_and(cv2.dilate(better_mask, small_kernel, iterations=1), strict_value_mask)

for i in range(200):
    best_mask = cv2.bitwise_and(cv2.dilate(best_mask, small_kernel, iterations=1), strict_value_mask)

best_mask = cv2.erode(cv2.dilate(best_mask, kernel, iterations = 2), kernel)

cv2.imshow("Mask", best_mask)

cv2.imwrite("zebra_segmentado.jpg", best_mask)

#Dice score

segmentado = cv2.imread("zebra_segmentado.jpg")

ground_truth = cv2.imread("ground_truth_zebra.jpg")

dice_score = cv2.bitwise_not(cv2.bitwise_xor(segmentado, ground_truth))

area_correta = np.sum(dice_score == 255)

area_total = np.sum(dice_score == 255) + np.sum(dice_score == 0)

print(area_correta/area_total)
