import config as cfg
import os

with open(os.path.join(cfg.data_root, "train_gt.txt"), "r") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    li = line.split()
    for i in range(6, 10):
        li[i] = str(int(li[i]) + 1)
    new_line = " ".join(li)
    new_lines.append(new_line)

new_file = "\n".join(new_lines) 

with open(os.path.join(cfg.data_root, "new_train_gt.txt"), "w") as f:
    f.write(new_file)