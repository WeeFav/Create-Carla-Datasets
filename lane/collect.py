import carla
import random
import cv2
import numpy as np
import sys
import pygame

import config as cfg
from carla_sync_mode import CarlaSyncMode
from lane_markings import LaneMarkings


# ==============================================================================
# -- Carla Game ---------------------------------------------------------------
# ==============================================================================

class CarlaGame():
    """
    Main Game Instance to execute carla simulator in pygame.
    """ 
    def __init__(self):
        self.display = pygame.display.set_mode((cfg.image_width, cfg.image_height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.pygame_clock = pygame.time.Clock()

        self.client = carla.Client('localhost', 2000)
        self.world = self.client.get_world()
        self.map = self.world.get_map()

        # Traffic manager
        self.tm = self.client.get_trafficmanager()
        self.tm.set_hybrid_physics_mode(True)

        blueprint_library = self.world.get_blueprint_library()
        spawn_points = self.world.get_map().get_spawn_points()

        # Spawn ego vehicle
        bp_ego_vehicle = random.choice(blueprint_library.filter('vehicle.ford.mustang'))
        bp_ego_vehicle.set_attribute('role_name', 'hero')
        self.ego_vehicle = self.world.spawn_actor(bp_ego_vehicle, random.choice(spawn_points))
        self.ego_vehicle.set_autopilot(True, self.tm.get_port())

        # Spawn rgb-cam and attach to vehicle
        bp_camera_rgb = blueprint_library.find('sensor.camera.rgb')
        bp_camera_rgb.set_attribute('image_size_x', f'{cfg.image_width}')
        bp_camera_rgb.set_attribute('image_size_y', f'{cfg.image_height}')
        bp_camera_rgb.set_attribute('fov', f'{cfg.fov}')
        camera_spawnpoint_rgb = carla.Transform(carla.Location(x=0.0, z=3.2), carla.Rotation(pitch=-19.5))
        self.camera_rgb = self.world.spawn_actor(bp_camera_rgb, camera_spawnpoint_rgb, attach_to=self.ego_vehicle)

        # Spawn semseg-cam and attach to vehicle
        bp_camera_semseg = blueprint_library.find('sensor.camera.semantic_segmentation')
        bp_camera_semseg.set_attribute('image_size_x', f'{cfg.image_width}')
        bp_camera_semseg.set_attribute('image_size_y', f'{cfg.image_height}')
        bp_camera_semseg.set_attribute('fov', f'{cfg.fov}')
        camera_semseg_spawnpoint = carla.Transform(carla.Location(x=0.0, z=3.2), carla.Rotation(pitch=-19.5))
        self.camera_semseg = self.world.spawn_actor(bp_camera_semseg, camera_semseg_spawnpoint, attach_to=self.ego_vehicle)

        # Spawn vehicles
        vehicle_blueprints = blueprint_library.filter('*vehicle*')

        i = 0
        while i < cfg.num_vehicles:
            vehicle = self.world.try_spawn_actor(random.choice(vehicle_blueprints), random.choice(spawn_points))
            if vehicle is not None:
                vehicle.set_autopilot(True, self.tm.get_port())
                i += 1
            
        self.lanemarkings = LaneMarkings(self.client)


    def reshape_image(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        return array


    def draw_image(self, surface, image, blend=False):
        array = self.reshape_image(image)
        image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if blend:
            image_surface.set_alpha(100)
        surface.blit(image_surface, (0, 0))


    def render_display(self, display, image, image_semseg, lanes_list, colormap, render_lanes=True):
        """
        Renders the images captured from both cameras and shows it on the
        pygame display

        Args:
            image: numpy array. Shows the 3-channel numpy imagearray on the pygame display.
            image_semseg: numpy array. Shows the semantic segmentation image on the pygame display.
        """
        # Draw the display.
        self.draw_image(display, image)
        #draw_image(display, image_semseg, blend=True)
        
        # Draw lanepoints of every lane on pygame window
        if(render_lanes):
            for i, color in enumerate(colormap):
                if(lanes_list[i]):
                    for j in range(len(lanes_list[i])):
                        pygame.draw.circle(display, colormap[color], lanes_list[i][j], 3, 2)
            
        pygame.display.flip()


    def should_quit(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
        return False
    

    def run(self):
        with CarlaSyncMode(self.world, self.tm, self.camera_rgb, self.camera_semseg, fps=cfg.fps) as sync_mode:            
            while True:
                    if self.should_quit():
                        break

                    # clock tick
                    self.pygame_clock.tick()
                    snapshot, image_rgb, image_semseg = sync_mode.tick(timeout=1.0)

                    # Get current waypoints
                    waypoint = self.map.get_waypoint(self.ego_vehicle.get_location())
                    waypoint_list = []
                    for i in range(0, cfg.number_of_lanepoints):
                        waypoint_list.append(waypoint.next(i + cfg.meters_per_frame)[0])
                    
                    # Convert and reshape image from Nx1 to shape(720, 1280, 3)
                    image_semseg.convert(carla.ColorConverter.CityScapesPalette)
                    image_semseg = self.reshape_image(image_semseg)
                    
                    # Calculate lanepoints for all lanes
                    lanes_list, x_lanes_list = self.lanemarkings.detect_lanemarkings(waypoint_list, image_semseg, self.camera_rgb)

                    for waypoint in waypoint_list:
                        self.world.debug.draw_point(location=waypoint.transform.location, size=0.05, life_time=cfg.number_of_lanepoints/cfg.fps, persistent_lines=False)                    
                    
                    # Show lanes on pygame
                    self.render_display(self.display, image_rgb, image_semseg, lanes_list, self.lanemarkings.colormap, cfg.render_lanes)

            # Destroy all actors after game ends
            for vehicle in self.world.get_actors().filter('*vehicle*'):
                vehicle.destroy()
            print("All vehicles destroyed")
            self.camera_rgb.destroy()
            self.camera_semseg.destroy()
            print("Cameras destroyed")

if __name__ == '__main__':
    pygame.init()

    game = CarlaGame()
    game.run()

    pygame.quit()






