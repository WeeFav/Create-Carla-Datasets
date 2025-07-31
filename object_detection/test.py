import numpy as np

def generate_3d_bbox_corners(bottom_center, dims, rotation_z):
    """
    Generate 8 corners of a 3D bounding box in LiDAR frame.

    Args:
        bottom_center (3,): [x, y, z] of the bottom center in LiDAR frame
        dims (3,): [height, width, length] of the object
        rotation_z: yaw rotation (rad) around LiDAR z-axis

    Returns:
        corners_lidar: (8, 3) array of 3D bounding box corners in LiDAR frame
                       order: [FLB, FRB, RRB, RLB, FLT, FRT, RRT, RLT]
    """
    h, w, l = dims

    # --- 1️⃣ Define corners in the object's local frame ---
    # Origin: bottom center
    # x-axis: forward, y-axis: left, z-axis: up
    x_corners = np.array([ l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2])
    y_corners = np.array([ w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2])
    z_corners = np.array([ 0,    0,    0,    0,    h,    h,    h,    h   ])

    corners_obj = np.vstack((x_corners, y_corners, z_corners))  # (3, 8)

    # --- 2️⃣ Rotation matrix around Z axis ---
    cos_yaw = np.cos(rotation_z)
    sin_yaw = np.sin(rotation_z)
    R = np.array([
        [cos_yaw, -sin_yaw, 0],
        [sin_yaw,  cos_yaw, 0],
        [0,        0,       1]
    ])

    # --- 3️⃣ Rotate and translate to LiDAR frame ---
    corners_lidar = (R @ corners_obj).T + bottom_center  # (8,3)

    return corners_lidar


# =====================
# Example usage:
# =====================
bottom_center = np.array([10.0, 5.0, 0.2])   # LiDAR coords
dims = np.array([1.5, 2.0, 4.0])             # height, width, length
rotation_z = np.deg2rad(30)                   # 30 degrees

corners = generate_3d_bbox_corners(bottom_center, dims, rotation_z)
print("Corners in LiDAR frame:\n", corners)
