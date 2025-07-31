import carla
import random
import cv2
import numpy as np
import sys
import pygame
import os
import argparse
from PIL import Image
import time
import math
import open3d as o3d

import config as cfg
from carla_sync_mode import CarlaSyncMode
from vehicle_manager import VehicleManager
import utils

class CarlaGame():
    def __init__(self):
        self.display = pygame.display.set_mode((cfg.image_width, cfg.image_height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.pygame_clock = pygame.time.Clock()

        self.client = carla.Client('localhost', 2000)
        self.world = self.client.load_world(cfg.town)
        self.map = self.world.get_map()
        self.world.set_weather(cfg.weather)
        self.tm = self.client.get_trafficmanager()

        self.vehicle_manager = VehicleManager(self.client, self.world, self.tm)
        self.ego_vehicle = self.vehicle_manager.spawn_ego_vehicle()


        # Get the bounding box
        bbox = self.ego_vehicle.bounding_box
        # Vehicle dimensions
        length = bbox.extent.x * 2  # CARLA gives half-length
        width = bbox.extent.y * 2
        height = bbox.extent.z * 2
        print(f"Vehicle dimensions: Length={length:.2f}m, Width={width:.2f}m, Height={height:.2f}m")

        blueprint_library = self.world.get_blueprint_library()
        # Spawn rgb-cam and attach to vehicle
        bp_camera_rgb = blueprint_library.find('sensor.camera.rgb')
        bp_camera_rgb.set_attribute('image_size_x', f'{cfg.image_width}')
        bp_camera_rgb.set_attribute('image_size_y', f'{cfg.image_height}')
        bp_camera_rgb.set_attribute('fov', f'{cfg.fov}')
        self.camera_spawnpoint = carla.Transform(carla.Location(x=1.0, z=1.65), carla.Rotation(pitch=-18.5)) # camera 5
        self.camera_rgb = self.world.spawn_actor(bp_camera_rgb, self.camera_spawnpoint, attach_to=self.ego_vehicle)

        # Spawn semseg-cam and attach to vehicle
        bp_camera_semseg = blueprint_library.find('sensor.camera.semantic_segmentation')
        bp_camera_semseg.set_attribute('image_size_x', f'{cfg.image_width}')
        bp_camera_semseg.set_attribute('image_size_y', f'{cfg.image_height}')
        bp_camera_semseg.set_attribute('fov', f'{cfg.fov}')
        self.camera_semseg = self.world.spawn_actor(bp_camera_semseg, self.camera_spawnpoint, attach_to=self.ego_vehicle)

        # Spawn depth-cam and attach to vehicle
        bp_camera_depth = blueprint_library.find('sensor.camera.depth')
        bp_camera_depth.set_attribute('image_size_x', f'{cfg.image_width}')
        bp_camera_depth.set_attribute('image_size_y', f'{cfg.image_height}')
        bp_camera_depth.set_attribute('fov', f'{cfg.fov}')
        self.camera_depth = self.world.spawn_actor(bp_camera_depth, self.camera_spawnpoint, attach_to=self.ego_vehicle)
        
        # Spawn lidar and attach to vehicle
        bp_lidar = blueprint_library.find("sensor.lidar.ray_cast")
        bp_lidar.set_attribute("range", "120") # 120 meter range for cars and foliage
        bp_lidar.set_attribute("rotation_frequency", "20") # 20 match carla simulation fps
        bp_lidar.set_attribute("channels", "64") # vertical resolution of the laser scanner is 64
        bp_lidar.set_attribute("points_per_second", "1000000") # 100k points per cycle
        bp_lidar.set_attribute("upper_fov", "2.0") # +2 up to -24.8 down
        bp_lidar.set_attribute("lower_fov", "-24.8")
        self.lidar_spawnpoint = carla.Transform(carla.Location(x=0, y=0, z=1.73))
        self.lidar = self.world.spawn_actor(bp_lidar, self.lidar_spawnpoint, attach_to=self.ego_vehicle)

        self.vehicle_manager.spawn_vehicles()

        self.tick_counter = 0

        self.RGB_colors = [(0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 0, 255)]
        self.BGR_colors = [(0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 0)]


        # Initialize Open3D Visualizer
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name='CARLA LiDAR', width=800, height=600)
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(np.random.rand(10, 3))
        self.vis.add_geometry(self.pcd)

        render_opt = self.vis.get_render_option()
        render_opt.background_color = np.asarray([0, 0, 0])
        render_opt.point_size = 1
        
        ctr = self.vis.get_view_control()
        ctr.change_field_of_view(step=90)
        ctr.set_constant_z_far(2000)
        ctr.set_constant_z_near(0.1)
        self.vis.reset_view_point(True)
        self.cam = ctr.convert_to_pinhole_camera_parameters()

        self.R_carla_to_open3d = np.array([
            [0, 1, 0],
            [0, 0, -1],
            [1, 0, 0]
        ])

        self.bbox_lines = []

        # Define line connections between the 8 corners
        self.lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # bottom rectangle
            [4, 5], [5, 6], [6, 7], [7, 4],  # top rectangle
            [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
        ]


    def reshape_image(self, sensor):
        array = np.frombuffer(sensor.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (sensor.height, sensor.width, 4)) # BGRA
        array = array[:, :, :3] # BGR
        array = array[:, :, ::-1] # RGB
        return array # (H, W, C)
    
    
    def reshape_pointcloud(self, pointcloud):
        array = np.frombuffer(pointcloud.raw_data, dtype=np.float32)
        array = np.reshape(array, (-1, 4)) # x, y, z, r
        return array # (N, 4) pointcloud       


    def draw_image(self, surface, array, blend=False):
        image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1)) # (W, H, C)
        if blend:
            image_surface.set_alpha(100)
        surface.blit(image_surface, (0, 0))
        
    
    def get_calib(self):
        # Intrinsic Matrix (K)
        self.f = cfg.image_width / (2 * math.tan(cfg.fov * math.pi / 360))
        self.c_x = cfg.image_width/2
        self.c_y = cfg.image_height/2
        
        K = np.float32([[self.f, 0, self.c_x],
                        [0, self.f, self.c_y],
                        [0, 0, 1]])        
        
        # Extrinsic Matrix (R|T)
        T_world_to_cam = np.array(self.camera_rgb.get_transform().get_inverse_matrix())
        RT = T_world_to_cam[0:3, :] # (3, 4)
        
        # Projection Matrix
        P = K @ RT  # (3, 4)
        
        # Lidar to world
        T_lidar_to_world = np.array(self.lidar.get_transform().get_matrix())
        # World to camera
        T_world_to_camera = np.array(self.camera.get_transform().get_inverse_matrix())
        # Lidar to camera
        Tr_lidar_to_cam = T_world_to_camera @ T_lidar_to_world # (4, 4)
        Tr_velo_to_cam = Tr_lidar_to_cam[0:3, :] # (3, 4)
        
        # R0_rect: 3×3 rectification matrix (identity for monocular setup)
        R0_rect = np.identity(3)
        
        return P, Tr_velo_to_cam, R0_rect
    

    def render_display(self, image_rgb, pointcloud, bboxes):
        # Draw the display.
        self.draw_image(self.display, image_rgb)

        pointcloud = pointcloud @ self.R_carla_to_open3d.T

        # Update point cloud
        self.pcd.points = o3d.utility.Vector3dVector(pointcloud)
        self.pcd.colors = o3d.utility.Vector3dVector(np.tile([1.0, 1.0, 0.0], (pointcloud.shape[0], 1)))
        self.vis.update_geometry(self.pcd)

        # Draw Bounding Boxes
        # Clear previous bounding boxes
        for line in self.bbox_lines:
            self.vis.remove_geometry(line, reset_bounding_box=False)
        self.bbox_lines = []

        bboxes_corners = [bbox['corners_lidar'] for bbox in bboxes]
        for corners in bboxes_corners:
            # Apply transformation to Open3D coordinate frame
            corners = corners @ self.R_carla_to_open3d.T

            # Create LineSet
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(corners)
            line_set.lines = o3d.utility.Vector2iVector(self.lines)

            # Set green color for all lines
            colors = [[0.0, 1.0, 0.0] for _ in range(len(self.lines))]
            line_set.colors = o3d.utility.Vector3dVector(colors)

            # Add to visualizer and keep reference
            self.vis.add_geometry(line_set)
            self.bbox_lines.append(line_set)


        self.vis.get_view_control().convert_from_pinhole_camera_parameters(self.cam)
        self.vis.poll_events()
        self.vis.update_renderer()
        self.cam = self.vis.get_view_control().convert_to_pinhole_camera_parameters()

        pygame.display.flip()


    def run(self):
        with CarlaSyncMode(self.world, self.tm, self.camera_rgb, self.camera_semseg, self.camera_depth, self.lidar, fps=cfg.fps) as sync_mode:
            try:
                while True:
                    ### pygame interaction ###
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            exit()
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_LEFT:
                                print("left")
                                self.tm.force_lane_change(self.ego_vehicle, False)
                            elif event.key == pygame.K_RIGHT:
                                print("right")
                                self.tm.force_lane_change(self.ego_vehicle, True)
                    

                    ### Manually run simulation ###
                    if not cfg.auto_run:
                        waiting = True
                        while waiting:
                            event = pygame.event.wait()
                            if event.type == pygame.QUIT:
                                pygame.quit()
                                exit()
                            keys = pygame.key.get_pressed()
                            if keys[pygame.K_SPACE]:
                                waiting = False


                    ### Respawn stuck vehicle ###
                    if self.tick_counter % (cfg.respawn * cfg.fps) == 0:
                        self.vehicle_manager.check_vehicle()
                

                    ### Run simulation ###
                    self.pygame_clock.tick()
                    snapshot, sensor_rgb, sensor_semseg, sensor_depth, sensor_lidar = sync_mode.tick(timeout=1.0)
                    image_rgb = self.reshape_image(sensor_rgb)
                    image_depth = self.reshape_image(sensor_depth)
                    sensor_semseg.convert(carla.ColorConverter.CityScapesPalette)
                    image_semseg = self.reshape_image(sensor_semseg)
                    pointcloud = self.reshape_pointcloud(sensor_lidar)
                    self.tick_counter += 1

                    bboxes = utils.get_bboxes(self.world, pointcloud[:, :3], sensor_lidar)

                    ### Render display ###
                    self.render_display(image_rgb, pointcloud[:, :3], bboxes)

            finally:
                self.vehicle_manager.destroy()
                self.camera_rgb.destroy()
                self.camera_semseg.destroy()
                self.camera_depth.destroy()
                self.lidar.destroy()
                print("Cameras destroyed")


if __name__ == '__main__':
    pygame.init()

    game = CarlaGame()
    game.run()

    pygame.quit()
