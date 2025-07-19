import cv2
from PIL import Image
import numpy as np

img = cv2.imread("./CULane/3_img.png")
label = np.asarray(Image.open("./CULane/3_label.png"))

print(img.shape)
print(label.shape)

np.savetxt("./CULane/vis.txt", label, fmt="%d")

for r in range(label.shape[0]):
    for c in range(label.shape[1]):
        if label[r][c] == 70:
            img[r][c] = (255, 0, 0)
        elif label[r][c] == 120:
            img[r][c] = (0, 255, 0)
        elif label[r][c] == 20:
            img[r][c] = (0, 0, 255)
        elif label[r][c] == 170:
            img[r][c] = (255, 255, 0)

cv2.imshow("vis", img)
cv2.waitKey(0)
cv2.destroyAllWindows() 