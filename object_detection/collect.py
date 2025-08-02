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
    def __init__(self, run_name):
        self.display = pygame.display.set_mode((cfg.image_width, cfg.image_height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.pygame_clock = pygame.time.Clock()

        self.client = carla.Client('localhost', 2000)
        self.world = self.client.load_world(cfg.town)
        self.map = self.world.get_map()
        self.world.set_weather(cfg.weather)
        self.tm = self.client.get_trafficmanager()

        self.vehicle_manager = VehicleManager(self.client, self.world, self.tm)
        self.ego_vehicle = self.vehicle_manager.spawn_ego_vehicle()

        blueprint_library = self.world.get_blueprint_library()
        # Spawn rgb-cam and attach to vehicle
        bp_camera_rgb = blueprint_library.find('sensor.camera.rgb')
        bp_camera_rgb.set_attribute('image_size_x', f'{cfg.image_width}')
        bp_camera_rgb.set_attribute('image_size_y', f'{cfg.image_height}')
        bp_camera_rgb.set_attribute('fov', f'{cfg.fov}')
        self.camera_spawnpoint = carla.Transform(carla.Location(x=1.0, z=1.65), carla.Rotation(pitch=-18.5)) # camera 5
        self.camera_rgb = self.world.spawn_actor(bp_camera_rgb, self.camera_spawnpoint, attach_to=self.ego_vehicle)
        
        # Spawn lidar and attach to vehicle
        bp_lidar = blueprint_library.find("sensor.lidar.ray_cast")
        bp_lidar.set_attribute("range", "120") # 120 meter range for cars and foliage
        bp_lidar.set_attribute("rotation_frequency", "10")
        bp_lidar.set_attribute("channels", "64") # vertical resolution of the laser scanner is 64
        bp_lidar.set_attribute("points_per_second", "1300000")
        bp_lidar.set_attribute("upper_fov", "2.0") # +2 up to -24.8 down
        bp_lidar.set_attribute("lower_fov", "-24.8")
        self.lidar_spawnpoint = carla.Transform(carla.Location(x=0, y=0, z=1.73))
        self.lidar = self.world.spawn_actor(bp_lidar, self.lidar_spawnpoint, attach_to=self.ego_vehicle)

        self.vehicle_manager.spawn_vehicles()

        self.tick_counter = 0

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

        # Create save path
        if cfg.saving:
            if run_name:
                run_root = os.path.join(cfg.data_root, f"{cfg.town}_{run_name}")
            else:
                run_root = os.path.join(cfg.data_root, f"{cfg.town}")
            self.calib_folder = os.path.join(run_root, "calib")
            self.image_folder = os.path.join(run_root, "image_2")
            self.label_folder = os.path.join(run_root, "label_2")
            self.velodyne_folder = os.path.join(run_root, "velodyne")
            os.makedirs(run_root, exist_ok=False)
            os.makedirs(self.calib_folder, exist_ok=True)
            os.makedirs(self.image_folder, exist_ok=True)
            os.makedirs(self.label_folder, exist_ok=True)
            os.makedirs(self.velodyne_folder, exist_ok=True)
        
            self.save_interval = cfg.save_freq * cfg.fps
            self.save_counter = 0
            self.skip_counter = 0
            self.skip_interval = cfg.skip_at_traffic_light_interval


    def reshape_image(self, sensor):
        array = np.frombuffer(sensor.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (sensor.height, sensor.width, 4)) # BGRA
        array = array[:, :, :3] # BGR
        array = array[:, :, ::-1] # RGB
        return array # (H, W, C)
    

    def crop_to_kitti(self, image):
        target_h, target_w = 375, 1242
        H, W, _ = image.shape

        # Compute crop start indices (center crop)
        top = (H - target_h) // 2
        left = (W - target_w) // 2

        # Perform crop
        cropped = image[top:top + target_h, left:left + target_w, :]
        return cropped
    
    
    def reshape_pointcloud(self, pointcloud):
        array = np.frombuffer(pointcloud.raw_data, dtype=np.float32)
        array = np.reshape(array, (-1, 4)) # x, y, z, r
        return array # (N, 4) pointcloud       


    def draw_image(self, surface, array, blend=False):
        image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1)) # (W, H, C)
        if blend:
            image_surface.set_alpha(100)
        surface.blit(image_surface, (0, 0))
            

    def render_display(self, image_rgb, pointcloud, bboxes):
        # Draw the display.
        self.draw_image(self.display, image_rgb)

        pointcloud = pointcloud @ self.R_carla_to_open3d.T

        # Update point cloud
        self.pcd.points = o3d.utility.Vector3dVector(pointcloud)
        self.pcd.colors = o3d.utility.Vector3dVector(np.tile([1.0, 1.0, 0.0], (pointcloud.shape[0], 1)))
        self.vis.update_geometry(self.pcd)

        # Clear previous bounding boxes
        for line in self.bbox_lines:
            self.vis.remove_geometry(line, reset_bounding_box=False)
        self.bbox_lines = []

        # Draw Bounding Boxes
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


    def save(self, P, Tr_velo_to_cam, R0_rect, image_rgb, bboxes, pointcloud):
        ### calib ###
        with open(os.path.join(self.calib_folder, f"{self.save_counter}.txt"), 'w') as f:
            # Flatten row-major and write each matrix            
            P2_flat = ' '.join(map(str, P.flatten()))
            f.write(f"P2: {P2_flat}\n")
            
            R0_flat = ' '.join(map(str, R0_rect.flatten()))
            f.write(f"R0_rect: {R0_flat}\n")
            
            Tr_flat = ' '.join(map(str, Tr_velo_to_cam.flatten()))
            f.write(f"Tr_velo_to_cam: {Tr_flat}\n")


        ### image ###
        image_rgb = self.crop_to_kitti(image_rgb)
        image_rgb = Image.fromarray(image_rgb)
        image_rgb.save(os.path.join(self.image_folder, f"{self.save_counter}.png"))


        ### label ###
        with open(os.path.join(self.label_folder, f"{self.save_counter}.txt"), 'w') as f:
            for bbox in bboxes:
                object_type = bbox['object_type']
                truncation = 0
                occlusion = 0
                alpha = 0
                left = 0
                top = 0
                right = 0
                bottom = 0
                height = bbox['dims'][0]
                width = bbox['dims'][1]
                length = bbox['dims'][2]
                x = bbox['bottom_center'][0]
                y = bbox['bottom_center'][1]
                z = bbox['bottom_center'][2]
                rotation_y = bbox['rotation_z']
                label_flat = ' '.join(map(str, [object_type, truncation, occlusion, alpha, left, top, right, bottom, height, width, length, x, y, z, rotation_y]))
                f.write(f"{label_flat}\n")
        
        
        ### velodyne ###
        pointcloud = pointcloud.astype(np.float32)
        pointcloud.tofile(os.path.join(self.velodyne_folder, f"{self.save_counter}.bin"))


    def run(self):
        with CarlaSyncMode(self.world, self.tm, self.camera_rgb, self.lidar, fps=cfg.fps) as sync_mode:
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
                    snapshot, sensor_rgb, sensor_lidar = sync_mode.tick(timeout=1.0)
                    image_rgb = self.reshape_image(sensor_rgb)
                    pointcloud = self.reshape_pointcloud(sensor_lidar)
                    self.tick_counter += 1

                    ### Get labels and calib ###
                    bboxes = utils.get_bboxes(self.world, pointcloud[:, :3], sensor_lidar)
                    P, Tr_velo_to_cam, R0_rect = utils.get_calib(self.camera_rgb, self.lidar)


                    ### Render display ###
                    self.render_display(image_rgb, pointcloud[:, :3], bboxes)


                    ### Save ###
                    if cfg.saving:
                        if self.tick_counter % self.save_interval == 0:
                            if self.ego_vehicle.is_at_traffic_light() or self.vehicle_manager.get_vehicle_state(self.ego_vehicle) == self.vehicle_manager.vehicle_state[self.ego_vehicle.id]:
                                if self.skip_counter % self.skip_interval != 0:
                                    self.skip_counter += 1
                                    print("save skipped at traffic")
                                    return
                                else:
                                    self.skip_counter += 1
                            else:
                                self.skip_counter = 0 # reset skip counter

                            self.vehicle_manager.vehicle_state[self.ego_vehicle.id] = self.vehicle_manager.get_vehicle_state(self.ego_vehicle)
                            self.save(P, Tr_velo_to_cam, R0_rect, image_rgb, bboxes, pointcloud)
                            print("saved:", self.save_counter)
                            self.save_counter += 1
                            if self.save_counter == cfg.save_num:
                                sys.exit()


            finally:
                self.vehicle_manager.destroy()
                self.camera_rgb.destroy()
                self.lidar.destroy()
                print("Cameras destroyed")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", required=False, default=None)
    args = parser.parse_args()

    pygame.init()

    game = CarlaGame(args.run_name)
    game.run()

    pygame.quit()
