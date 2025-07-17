import cv2
from PIL import Image
import numpy as np

img = cv2.imread("./CULane/01890_img.jpg")
label = np.asarray(Image.open("./CULane/01890_label.png"))

print(img.shape)
print(label.shape)

for r in range(label.shape[0]):
    for c in range(label.shape[1]):
        if label[r][c] == 1:
            img[r][c] = (255, 0, 0)
        elif label[r][c] == 2:
            img[r][c] = (0, 255, 0)
        elif label[r][c] == 3:
            img[r][c] = (0, 0, 255)
        elif label[r][c] == 4:
            img[r][c] = (255, 255, 0)

cv2.imshow("vis", img)
cv2.waitKey(0)
cv2.destroyAllWindows() 