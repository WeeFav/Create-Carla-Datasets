import os
import config as cfg

all_txt = []
for i in os.listdir(cfg.data_root):
    run_root = os.path.join(cfg.data_root, i)
    if os.path.isdir(run_root):
        txt_path = os.path.join(run_root, "train_gt.txt")
        all_txt.append(txt_path)
    
print(all_txt)

with open(os.path.join(cfg.data_root, "train_gt.txt"), "w") as f:
    for fname in all_txt:
        with open(fname, "r") as infile:
            f.write(infile.read())
        

