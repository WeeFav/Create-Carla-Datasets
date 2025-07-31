import numpy as np
import open3d as o3d

# Assume you already have:
# - self.pcd (PointCloud)
# - self.vis (Visualizer)
# - pointcloud: (N, 3)
# - bboxes: list of np.array (8, 3) each, for all bounding boxes

# Transform point cloud
pointcloud = pointcloud @ self.R_carla_to_open3d.T

# Update point cloud
self.pcd.points = o3d.utility.Vector3dVector(pointcloud)
self.pcd.colors = o3d.utility.Vector3dVector(
    np.tile([1.0, 1.0, 0.0], (pointcloud.shape[0], 1))
)
self.vis.update_geometry(self.pcd)

# ---- Draw Bounding Boxes ----
# Clear previous bounding boxes if necessary
if hasattr(self, "bbox_lines"):
    for line in self.bbox_lines:
        self.vis.remove_geometry(line, reset_bounding_box=False)
self.bbox_lines = []

# Define line connections between the 8 corners
lines = [
    [0, 1], [1, 2], [2, 3], [3, 0],  # bottom rectangle
    [4, 5], [5, 6], [6, 7], [7, 4],  # top rectangle
    [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
]

for corners in bboxes:
    # Apply transformation to Open3D coordinate frame
    corners_o3d = corners @ self.R_carla_to_open3d.T

    # Create LineSet
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners_o3d)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    # Set green color for all lines
    colors = [[0.0, 1.0, 0.0] for _ in range(len(lines))]
    line_set.colors = o3d.utility.Vector3dVector(colors)

    # Add to visualizer and keep reference
    self.vis.add_geometry(line_set)
    self.bbox_lines.append(line_set)

# Render updates
self.vis.poll_events()
self.vis.update_renderer()
