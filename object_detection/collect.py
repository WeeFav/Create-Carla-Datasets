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

import config as cfg
from carla_sync_mode import CarlaSyncMode
from vehicle_manager import VehicleManager


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

        blueprint_library = self.world.get_blueprint_library()
        # Spawn rgb-cam and attach to vehicle
        bp_camera_rgb = blueprint_library.find('sensor.camera.rgb')
        bp_camera_rgb.set_attribute('image_size_x', f'{cfg.image_width}')
        bp_camera_rgb.set_attribute('image_size_y', f'{cfg.image_height}')
        bp_camera_rgb.set_attribute('fov', f'{cfg.fov}')
        self.camera_spawnpoint = carla.Transform(carla.Location(x=1.0, z=2.0), carla.Rotation(pitch=-18.5)) # camera 5
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
        bp_lidar.set_attribute("rotation_frequency", "10") # 10 frames per second
        bp_lidar.set_attribute("channels", "64") # vertical resolution of the laser scanner is 64
        bp_lidar.set_attribute("points_per_second", "1000000") # 100k points per cycle
        bp_lidar.set_attribute("upper_fov", "2.0") # +2 up to -24.8 down
        bp_lidar.set_attribute("lower_fov", "-24.8")
        self.lidar_spawnpoint = carla.Transform(carla.Location(x=0, z=2.5))
        self.lidar = self.world.spawn_actor(bp_lidar, self.lidar_spawnpoint, attach_to=self.ego_vehicle)

        self.vehicle_manager.spawn_vehicles()

        self.tick_counter = 0

        self.RGB_colors = [(0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 0, 255)]
        self.BGR_colors = [(0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 0)]
        

    def reshape_image(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4)) # BGRA
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


    def render_display(self, image_rgb):
        # Draw the display.
        self.draw_image(self.display, image_rgb)
        
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
                    snapshot, image_rgb, image_semseg, image_depth, pointcloud = sync_mode.tick(timeout=1.0)
                    image_rgb = self.reshape_image(image_rgb)
                    image_depth = self.reshape_image(image_depth)
                    image_semseg.convert(carla.ColorConverter.CityScapesPalette)
                    image_semseg = self.reshape_image(image_semseg)
                    pointcloud = self.reshape_pointcloud(pointcloud)
                    self.tick_counter += 1
                    

                    ### Render display ###
                    self.render_display(image_rgb)

            finally:
                self.vehicle_manager.destroy()
                self.camera_rgb.destroy()
                self.camera_semseg.destroy()
                self.camera_depth.destroy()
                print("Cameras destroyed")


if __name__ == '__main__':
    pygame.init()

    game = CarlaGame()
    game.run()

    pygame.quit()
