import cv2
import json

img = cv2.imread("./TUSimple/360/20.jpg")

with open("./TUSimple/label_data_0313.json", 'r') as f:
    labels = json.load(f)

lanes = labels['lanes']
h_samples = labels['h_samples']
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]

for line, color in zip(lanes, colors):
    for x, y, in zip(line, h_samples):
        if x == -2:
            continue

        cv2.circle(img, (x, y), 2, color, 1)

cv2.imshow("vis", img)
cv2.waitKey(0)
cv2.destroyAllWindows() 