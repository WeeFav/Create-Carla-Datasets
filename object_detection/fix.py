import os
import numpy as np

kitti_root = "C:/Users/marvi/Datasets/Object/kitti"

training_calib = os.path.join(kitti_root, "training", "calib")
validating_calib = os.path.join(kitti_root, "validating", "calib")
testing_calib = os.path.join(kitti_root, "testing", "calib")

def add_imu_to_velo(calib_path):
    for file_name in os.listdir(calib_path):
        with open(os.path.join(calib_path, file_name), 'a') as f:
            PX = ' '.join(map(str, np.zeros(12)))
            f.write(f"Tr_imu_to_velo: {PX}")

add_imu_to_velo(training_calib)
add_imu_to_velo(validating_calib)
add_imu_to_velo(testing_calib)