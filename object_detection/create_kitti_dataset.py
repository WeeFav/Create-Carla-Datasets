import os
import numpy as np
from sklearn.model_selection import train_test_split

data_root = "C:/Users/marvi/Datasets/Object/CarlaKitti"
kitti_root = "C:/Users/marvi/Datasets/Object/kitti"

all_data = []
# calculate anchor sizes
car_dims = [] # (N, 3) where columns are w, l, h
large_dims = []
motorcycle_dims = []

for run_name in os.listdir(data_root):
    for filename in os.listdir(os.path.join(data_root, run_name, 'calib')):
        id = os.path.splitext(filename)[0]
        all_data.append(f"{run_name} {id}")

        with open(os.path.join(data_root, run_name, 'label_2', filename), 'r') as f:
            lines = f.readlines()
            for line in lines:
                label = line.split()
                object_type = label[0]
                height = float(label[8])
                width = float(label[9])
                length = float(label[10])
                if object_type == 'Car':
                    car_dims.append([width, length, height])
                elif object_type == 'Large':
                    large_dims.append([width, length, height])
                elif object_type == 'Motorcycle':
                    motorcycle_dims.append([width, length, height])

car_dims = np.array(car_dims)
large_dims = np.array(large_dims)
motorcycle_dims = np.array(motorcycle_dims)

car_anchor_size = car_dims.mean(axis=0)
large_anchor_size = large_dims.mean(axis=0)
motorcycle_anchor_size = motorcycle_dims.mean(axis=0)

print("Car anchor:", car_anchor_size)
print("Large anchor:", large_anchor_size)
print("Motorcycle anchor:", motorcycle_anchor_size)


trainval_data, test_data = train_test_split(list(range(len(all_data))), test_size=0.1)
train_data, val_data = train_test_split(trainval_data, test_size=0.15)
assert len(all_data) == len(train_data) + len(val_data) + len(test_data)

print("All data:", len(all_data))
print("Train:", len(train_data))
print("Val:", len(val_data))
print("Test:", len(test_data))

def move(split, split_idx):
    split_path = os.path.join(kitti_root, split)
    calib_folder = os.path.join(split_path, "calib")
    image_folder = os.path.join(split_path, "image_2")
    label_folder = os.path.join(split_path, "label_2")
    velodyne_folder = os.path.join(split_path, "velodyne")
    os.makedirs(split_path, exist_ok=True)
    os.makedirs(calib_folder, exist_ok=True)
    os.makedirs(image_folder, exist_ok=True)
    os.makedirs(label_folder, exist_ok=True)
    os.makedirs(velodyne_folder, exist_ok=True)

    split_idx_list = []

    for idx in split_idx:
        name = all_data[idx]
        run_name, id = name.split()

        source_path = os.path.join(data_root, run_name, "calib", f"{id}.txt")
        destination_path = os.path.join(calib_folder, f"{idx:06d}.txt")
        os.rename(source_path, destination_path)

        source_path = os.path.join(data_root, run_name, "image_2", f"{id}.png")
        destination_path = os.path.join(image_folder, f"{idx:06d}.png")
        os.rename(source_path, destination_path)

        source_path = os.path.join(data_root, run_name, "label_2", f"{id}.txt")
        destination_path = os.path.join(label_folder, f"{idx:06d}.txt")
        os.rename(source_path, destination_path)

        source_path = os.path.join(data_root, run_name, "velodyne", f"{id}.bin")
        destination_path = os.path.join(velodyne_folder, f"{idx:06d}.bin")
        os.rename(source_path, destination_path)

        split_idx_list.append(f"{idx:06d}")
    
    if split == "training":
        split_txt = "train"
    elif split == "validating":
        split_txt = "val"
    else:
        split_txt = "test"

    with open(os.path.join(kitti_root, f"{split_txt}.txt"), 'w') as f:
        f.write("\n".join(sorted(split_idx_list)))

move("training", train_data)
move("validating", val_data)
move("testing", test_data)
