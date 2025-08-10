import numpy as np
import open3d as o3d
import utils
import time


def bbox3d2corners(bboxes):
    '''
    bboxes: shape=(n, 7)
    return: shape=(n, 8, 3)
           ^ z   x            6 ------ 5
           |   /             / |     / |
           |  /             2 -|---- 1 |   
    y      | /              |  |     | | 
    <------|o               | 7 -----| 4
                            |/   o   |/    
                            3 ------ 0 
    x: front, y: left, z: top
    '''
    centers, dims, angles = bboxes[:, :3], bboxes[:, 3:6], bboxes[:, 6]
    dims = dims[:, [2, 1, 0]]

    # 1.generate bbox corner coordinates, clockwise from minimal point
    bboxes_corners = np.array([[-0.5, -0.5, 0], [-0.5, -0.5, 1.0], [-0.5, 0.5, 1.0], [-0.5, 0.5, 0.0],
                               [0.5, -0.5, 0], [0.5, -0.5, 1.0], [0.5, 0.5, 1.0], [0.5, 0.5, 0.0]], 
                               dtype=np.float32)
    bboxes_corners = bboxes_corners[None, :, :] * dims[:, None, :] # (1, 8, 3) * (n, 1, 3) -> (n, 8, 3)

    # 2. rotate around z axis
    rot_sin, rot_cos = np.sin(angles), np.cos(angles)
    # in fact, -angle
    rot_mat = np.array([[rot_cos, -rot_sin, np.zeros_like(rot_cos)],
                        [rot_sin, rot_cos, np.zeros_like(rot_cos)],
                        [np.zeros_like(rot_cos), np.zeros_like(rot_cos), np.ones_like(rot_cos)]], 
                        dtype=np.float32) # (3, 3, n)
    rot_mat = np.transpose(rot_mat, (2, 1, 0)) # (n, 3, 3)
    bboxes_corners = bboxes_corners @ rot_mat # (n, 8, 3)

    # 3. translate to centers
    bboxes_corners += centers[:, None, :]
    return bboxes_corners


def mybbox3d2corners(bboxes):
    # bottom_center = np.asarray(bboxes['bottom_center'])  # (N, 3)
    # h, w, l = np.split(bboxes['dims'], 3, axis=1)        # (N, 1) each
    # rotation_z = np.asarray(bboxes['rotation_z'])        # (N,)

    bottom_center, dims, rotation_z = bboxes[:, :3], bboxes[:, 3:6], bboxes[:, 6]
    dims = dims[:, [2, 1, 0]]

    # Base corners in local bbox frame (no rotation, centered at origin)
    base_corners = np.array([
        [ 0.5, -0.5, 0.0],  # front-left-bottom
        [ 0.5,  0.5, 0.0],  # front-right-bottom
        [-0.5,  0.5, 0.0],  # rear-right-bottom
        [-0.5, -0.5, 0.0],  # rear-left-bottom
        [ 0.5, -0.5, 1.0],  # front-left-top
        [ 0.5,  0.5, 1.0],  # front-right-top
        [-0.5,  0.5, 1.0],  # rear-right-top
        [-0.5, -0.5, 1.0],  # rear-left-top
    ])  # (8, 3)

    # Scale base corners by l, w, h for each box
    scaled_corners = base_corners[None, :, :] * dims[:, None, :]  # (N, 8, 3)

    # Rotation matrices for each bbox (around Z-axis)
    cos_yaw = np.cos(rotation_z)
    sin_yaw = np.sin(rotation_z)
    R = np.stack([
        np.stack([cos_yaw, -sin_yaw, np.zeros_like(cos_yaw)], axis=-1),
        np.stack([sin_yaw,  cos_yaw, np.zeros_like(cos_yaw)], axis=-1),
        np.stack([np.zeros_like(cos_yaw), np.zeros_like(cos_yaw), np.ones_like(cos_yaw)], axis=-1)
    ], axis=1)  # (N, 3, 3)

    # Rotate: (N, 8, 3) = (N, 8, 3) @ (N, 3, 3)^T
    rotated_corners = scaled_corners @ R.transpose(0, 2, 1)  # (N, 8, 3)

    # Translate
    corners_lidar = rotated_corners + bottom_center[:, None, :]  # (N, 8, 3)

    return corners_lidar






# Initialize Open3D Visualizer
vis = o3d.visualization.Visualizer()
vis.create_window(window_name='CARLA LiDAR', width=800, height=600)
pcd = o3d.geometry.PointCloud()

render_opt = vis.get_render_option()
render_opt.background_color = np.asarray([0, 0, 0])
render_opt.point_size = 1

ctr = vis.get_view_control()
ctr.change_field_of_view(step=90)
ctr.set_constant_z_far(2000)
ctr.set_constant_z_near(0.1)
vis.reset_view_point(True)
cam = ctr.convert_to_pinhole_camera_parameters()

bbox_lines = []

# Define line connections between the 8 corners
lines = [
    [0, 1], [1, 2], [2, 3], [3, 0],  # bottom rectangle
    [4, 5], [5, 6], [6, 7], [7, 4],  # top rectangle
    [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
]


mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
vis.add_geometry(mesh_frame)

pointcloud = np.fromfile(r"C:\Users\marvi\Datasets\Object\kitti\training\velodyne\000217.bin", dtype=np.float32)
pointcloud = pointcloud.reshape(-1, 4)[:, :3]
pointcloud[:, 1] = -pointcloud[:, 1] # convert from UE to Kitti/Open3D

pcd.points = o3d.utility.Vector3dVector(pointcloud)
pcd.colors = o3d.utility.Vector3dVector(np.tile([1.0, 1.0, 0.0], (pointcloud.shape[0], 1)))
vis.add_geometry(pcd)

with open(r"C:\Users\marvi\Datasets\Object\kitti\training\label_2\000217.txt") as f:
    labels = f.readlines()

bboxes = []
for label in labels:
    label = label.split()
    dims = np.array(list(map(float, label[8:11])))
    bottom_center = np.array(list(map(float, label[11:14])))
    rotation_z = np.array([label[14]], np.float64)
    bboxes.append(np.concatenate([bottom_center, dims, rotation_z]))
bboxes = np.array(bboxes) # (N, 7)

# Draw Bounding Boxes
bboxes_corners = bbox3d2corners(bboxes)

for corners in bboxes_corners:
    corners[:, 1] = -corners[:, 1] # convert from UE to Kitti/Open3D

    # Create LineSet
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    # Set green color for all lines
    colors = [[0.0, 1.0, 0.0] for _ in range(len(lines))]
    line_set.colors = o3d.utility.Vector3dVector(colors)

    # Add to visualizer and keep reference
    vis.add_geometry(line_set)


while True:
    vis.poll_events()
    vis.update_renderer()

