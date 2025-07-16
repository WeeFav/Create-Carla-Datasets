import cv2

img = cv2.imread("./CULane/02370.jpg")

with open("./CULane/02370.lines.txt", 'r') as f:
    lines = f.readlines()

colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
for line, color in zip(lines, colors):
    coords = line.split()
    i = 0
    while i < len(coords):
        x = int(float(coords[i]))
        y = int(float(coords[i + 1]))
        i += 2
        cv2.circle(img, (x, y), 2, color, 1)

cv2.imshow("vis", img)
cv2.waitKey(0)
cv2.destroyAllWindows()