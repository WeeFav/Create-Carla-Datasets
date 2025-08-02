import numpy as np
import open3d as o3d
import utils
import time

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

R_carla_to_open3d = np.array([
    [0, 1, 0],
    [0, 0, -1],
    [1, 0, 0]
])

bbox_lines = []

# Define line connections between the 8 corners
lines = [
    [0, 1], [1, 2], [2, 3], [3, 0],  # bottom rectangle
    [4, 5], [5, 6], [6, 7], [7, 4],  # top rectangle
    [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
]


pointcloud = np.fromfile("C:/Users/marvi/Datasets/Object/CarlaKitti/Town10HD_Opt/velodyne/6.bin", dtype=np.float32)
pointcloud = pointcloud.reshape(-1, 4)[:, :3]

pointcloud = pointcloud @ R_carla_to_open3d.T
pcd.points = o3d.utility.Vector3dVector(pointcloud)
pcd.colors = o3d.utility.Vector3dVector(np.tile([1.0, 1.0, 0.0], (pointcloud.shape[0], 1)))
vis.add_geometry(pcd)

with open("C:/Users/marvi/Datasets/Object/CarlaKitti/Town10HD_Opt/label_2/6.txt") as f:
    labels = f.readlines()

bboxes = []
for label in labels:
    label = label.split()
    object_type = label[0]
    dims = np.array(list(map(float, label[8:11])))
    bottom_center = np.array(list(map(float, label[11:14])))
    rotation_z = float(label[14])
    bboxes.append({
        'object_type': object_type,
        'dims': dims,
        'bottom_center': bottom_center,
        'rotation_z': rotation_z,
    })

# Draw Bounding Boxes
bboxes_corners = [utils.corners_from_bbox(bbox) for bbox in bboxes]

for corners in bboxes_corners:
    # Apply transformation to Open3D coordinate frame
    corners = corners @ R_carla_to_open3d.T

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

