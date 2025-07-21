import carla
import numpy as np
import cv2
from PIL import Image
import pygame
import random
import math
import sys

from carla_sync_mode import CarlaSyncMode
import config as cfg

class CarlaGame():
    """
    Main Game Instance to execute carla simulator in pygame.
    """ 
    def __init__(self):
        self.display = pygame.display.set_mode((cfg.image_width, cfg.image_height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.pygame_clock = pygame.time.Clock()

        self.client = carla.Client('localhost', 2000)
        self.world = self.client.load_world(cfg.town)
        self.map = self.world.get_map()
        self.tm = self.client.get_trafficmanager()

        blueprint_library = self.world.get_blueprint_library()
        spawn_points = self.world.get_map().get_spawn_points()

        # Spawn ego vehicle
        bp_ego_vehicle = random.choice(blueprint_library.filter('vehicle.ford.mustang'))
        bp_ego_vehicle.set_attribute('role_name', 'hero')
        self.ego_vehicle = self.world.spawn_actor(bp_ego_vehicle, random.choice(spawn_points))
        self.ego_vehicle.set_autopilot(True)

        # Spawn rgb-cam and attach to vehicle
        bp_camera_rgb = blueprint_library.find('sensor.camera.rgb')
        bp_camera_rgb.set_attribute('image_size_x', f'{cfg.image_width}')
        bp_camera_rgb.set_attribute('image_size_y', f'{cfg.image_height}')
        bp_camera_rgb.set_attribute('fov', f'{cfg.fov}')
        camera_spawnpoint = carla.Transform(carla.Location(x=1.0, z=2.0), carla.Rotation(pitch=-18.5)) # camera 5
        self.camera_rgb = self.world.spawn_actor(bp_camera_rgb, camera_spawnpoint, attach_to=self.ego_vehicle)

        # Spawn depth-cam and attach to vehicle
        bp_camera_depth = blueprint_library.find('sensor.camera.depth')
        bp_camera_depth.set_attribute('image_size_x', f'{cfg.image_width}')
        bp_camera_depth.set_attribute('image_size_y', f'{cfg.image_height}')
        bp_camera_depth.set_attribute('fov', f'{cfg.fov}')
        self.camera_depth = self.world.spawn_actor(bp_camera_depth, camera_spawnpoint, attach_to=self.ego_vehicle)
            
        self.colors = [[1,1,1], [2,2,2], [3,3,3], [4,4,4]]
        self.colors_display = [[70,70,70], [120,120,120], [20,20,20], [170,170,170]]

        # Intrinsic camera matrix needed to convert 3D-world coordinates to 2D-imagepoints
        self.f = cfg.image_width / (2 * math.tan(cfg.fov * math.pi / 360))
        self.c_x = cfg.image_width/2
        self.c_y = cfg.image_height/2
        
        self.cameraMatrix  = np.float32([[self.f, 0, self.c_x],
                                         [0, self.f, self.c_y],
                                         [0, 0, 1]])


    def should_quit(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
        return False
    

    def reshape_image(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        return array # (H, W, C)


    def draw_image(self, surface, array, blend=False):
        image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if blend:
            image_surface.set_alpha(100)
        surface.blit(image_surface, (0, 0))


    def calculate2DLanepoints(self, waypoint_list):
        world_points = []
        for waypoint in waypoint_list:
            world_points.append([waypoint.x, waypoint.y, waypoint.z, 1.0])
        world_points = np.float32(world_points).T
        
        # This (4, 4) matrix transforms the points from world to sensor coordinates.
        world_2_camera = np.array(self.camera_rgb.get_transform().get_inverse_matrix())
        
        # Transform the points from world space to camera space.
        sensor_points = np.dot(world_2_camera, world_points)
        
        # Now we must change from UE4's coordinate system to a "standard" one
        # (x, y ,z) -> (y, -z, x)
        point_in_camera_coords = np.array([sensor_points[1],
                                            sensor_points[2] * -1,
                                            sensor_points[0]])
        
        # Finally we can use our intrinsic matrix to do the actual 3D -> 2D.
        points_2d = np.dot(self.cameraMatrix, point_in_camera_coords)
        
        # Remember to normalize the x, y values by the 3rd value.
        points_2d = np.array([points_2d[0, :] / points_2d[2, :],
                                points_2d[1, :] / points_2d[2, :],
                                points_2d[2, :]])
        
        # visualize everything on a screen, the points that are out of the screen
        # must be discarted, the same with points behind the camera projection plane.
        points_2d = points_2d.T
        points_in_canvas_mask = (points_2d[:, 0] > 0.0) & (points_2d[:, 0] < cfg.image_width) & \
                                (points_2d[:, 1] > 0.0) & (points_2d[:, 1] < cfg.image_height) & \
                                (points_2d[:, 2] > 0.0)
        
        points_2d = points_2d[points_in_canvas_mask]
        
        # Extract the screen coords (xy) as integers.
        points_2d[:, 0] = points_2d[:, 0].astype(int)
        points_2d[:, 1] = points_2d[:, 1].astype(int)

        return points_2d[:, 0:2]

    def calculate_waypoint_from_2Dlane(self, point, image_depth):
        x = int(point[0])
        y = int(point[1])

        rgb = image_depth[y, x]
        R = rgb[0]
        G = rgb[1]
        B = rgb[2]

        normalized = (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1)
        depth_in_meters = 1000 * normalized

        # Convert 2D pixel to 3D camera standard coordinates
        x_std = (x - self.c_x) * depth_in_meters / self.f
        y_std = (y - self.c_y) * depth_in_meters / self.f
        z_std = depth_in_meters

        # Convert from standard camera to CARLA camera coordinates
        x_carla = z_std
        y_carla = x_std
        z_carla = -y_std

        # Convert to CARLA Location and transform to world space
        point_cam = carla.Location(x=x_carla, y=y_carla, z=z_carla)
        point_world = self.camera_depth.get_transform().transform(point_cam)
        return carla.Location(x=point_world.x, y=point_world.y, z=point_world.z)
    

    def calculate_waypoint_from_2Dlane_batch(self, points_2d, image_depth):
        x_coords = points_2d[:, 0].astype(int) # (N,)
        y_coords = points_2d[:, 1].astype(int)

        rgb = image_depth[y_coords, x_coords] # (N, 3)
        R = rgb[:, 0].astype(int) # (N,)
        G = rgb[:, 1].astype(int)
        B = rgb[:, 2].astype(int)

        normalized = (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1)
        depth_in_meters = 1000 * normalized

        # Convert 2D pixel to 3D camera standard coordinates
        x_std = (x_coords - self.c_x) * depth_in_meters / self.f
        y_std = (y_coords - self.c_y) * depth_in_meters / self.f
        z_std = depth_in_meters

        # Convert from standard camera to CARLA camera coordinates
        x_carla = z_std
        y_carla = x_std
        z_carla = -y_std

        # Convert to CARLA Location and transform to world space
        points_world = []
        for i in range(len(x_carla)):
            point_cam = carla.Location(x=x_carla[i], y=y_carla[i], z=z_carla[i])
            point_world = self.camera_depth.get_transform().transform(point_cam)
            points_world.append(carla.Location(x=point_world.x, y=point_world.y, z=point_world.z))

        return points_world


    def render_display(self, image_rgb, image_depth, waypoint_list):
        image_rgb = self.reshape_image(image_rgb) # (H, W, C)
        image_depth = self.reshape_image(image_depth) # (H, W, C)
        
        # draw rgb in pygame
        self.draw_image(self.display, image_rgb)
        
        # draw gt 3d point in carla
        for waypoint in waypoint_list:
            self.world.debug.draw_point(waypoint, size=0.05, life_time=2 * (1/cfg.fps), persistent_lines=False)    

        # draw 2d point in pygame
        self.tmp_list = []
        points_2d = self.calculate2DLanepoints(waypoint_list)
        for x, y, in points_2d:
            pygame.draw.circle(self.display, (0, 255, 0), (x, y), 3, 2)

        # draw converted 3d point in carla
        
        # sigle
        # for point in points_2d:
        #     w = self.calculate_waypoint_from_2Dlane(point, image_depth)
        #     self.world.debug.draw_point(w, size=0.05, color=carla.Color(255,255,0), life_time=2 * (1/cfg.fps), persistent_lines=False)    

        # batch
        w_list = self.calculate_waypoint_from_2Dlane_batch(points_2d, image_depth)
        for w in w_list:
            self.world.debug.draw_point(w, size=0.05, color=carla.Color(0,0,255), life_time=2 * (1/cfg.fps), persistent_lines=False)    

        pygame.display.flip()


    def run(self):
        with CarlaSyncMode(self.world, self.tm, self.camera_rgb, self.camera_depth, fps=cfg.fps) as sync_mode:
            try:
                while True:
                    if self.should_quit():
                        break

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

                    # clock tick
                    self.pygame_clock.tick()
                    snapshot, image_rgb, image_depth = sync_mode.tick(timeout=1.0)

                    # Get current waypoints
                    waypoint = self.map.get_waypoint(self.ego_vehicle.get_location())
                    waypoint_list = []
                    for i in range(0, cfg.number_of_lanepoints):
                        waypoint_list.append(waypoint.next(i + cfg.meters_per_frame)[0].transform.location)
                                        
                    # Show lanes on pygame
                    self.render_display(image_rgb, image_depth, waypoint_list)

            finally:
                # Destroy all actors after game ends
                for vehicle in self.world.get_actors().filter('*vehicle*'):
                    vehicle.destroy()
                print("All vehicles destroyed")
                self.camera_rgb.destroy()
                self.camera_depth.destroy()
                print("Cameras destroyed")

if __name__ == '__main__':
    game = CarlaGame()
    game.run()

