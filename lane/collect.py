import carla
import random
import cv2
import numpy as np
import sys
import pygame

import config as cfg
from carla_sync_mode import CarlaSyncMode
from lane_markings import LaneMarkings

def reshape_image(image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    return array

def draw_image(surface, image, blend=False):
    array = reshape_image(image)
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))

def render_display(display, image, image_semseg, lanes_list, colormap, render_lanes=True):
    """
    Renders the images captured from both cameras and shows it on the
    pygame display

    Args:
        image: numpy array. Shows the 3-channel numpy imagearray on the pygame display.
        image_semseg: numpy array. Shows the semantic segmentation image on the pygame display.
    """
    # Draw the display.
    draw_image(display, image)
    #draw_image(display, image_semseg, blend=True)
    
    # Draw lanepoints of every lane on pygame window
    if(render_lanes):
        for i, color in enumerate(colormap):
            if(lanes_list[i]):
                for j in range(len(lanes_list[i])):
                    pygame.draw.circle(display, colormap[color], lanes_list[i][j], 3, 2)
        
    pygame.display.flip()

def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
    return False

def main(client, world):
    pygame.init()
    display = pygame.display.set_mode((cfg.image_width, cfg.image_height), pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame_clock = pygame.time.Clock()

    map = world.get_map()

    blueprint_library = world.get_blueprint_library()

    start_position = random.choice(map.get_spawn_points())
    ego_vehicle = world.spawn_actor(random.choice(blueprint_library.filter('vehicle.ford.mustang')), start_position)
    ego_vehicle.set_autopilot(True)

    # Spawn rgb-cam and attach to vehicle
    bp_camera_rgb = blueprint_library.find('sensor.camera.rgb')
    bp_camera_rgb.set_attribute('image_size_x', f'{cfg.image_width}')
    bp_camera_rgb.set_attribute('image_size_y', f'{cfg.image_height}')
    bp_camera_rgb.set_attribute('fov', f'{cfg.fov}')
    camera_spawnpoint_rgb = carla.Transform(carla.Location(x=0.0, z=3.2), carla.Rotation(pitch=-19.5))
    camera_rgb = world.spawn_actor(bp_camera_rgb, camera_spawnpoint_rgb, attach_to=ego_vehicle)

    # Spawn semseg-cam and attach to vehicle
    bp_camera_semseg = blueprint_library.find('sensor.camera.semantic_segmentation')
    bp_camera_semseg.set_attribute('image_size_x', f'{cfg.image_width}')
    bp_camera_semseg.set_attribute('image_size_y', f'{cfg.image_height}')
    bp_camera_semseg.set_attribute('fov', f'{cfg.fov}')
    camera_semseg_spawnpoint = carla.Transform(carla.Location(x=0.0, z=3.2), carla.Rotation(pitch=-19.5))
    camera_semseg = world.spawn_actor(bp_camera_semseg, camera_semseg_spawnpoint, attach_to=ego_vehicle)

    vehicle_blueprints = blueprint_library.filter('*vehicle*')
    spawn_points = world.get_map().get_spawn_points()

    i = 0
    while i < 50:
        vehicle = world.try_spawn_actor(random.choice(vehicle_blueprints), random.choice(spawn_points))
        if vehicle is not None:
            vehicle.set_autopilot(True)
            i += 1
        
    lanemarkings = LaneMarkings(client)

    with CarlaSyncMode(world, camera_rgb, camera_semseg, fps=cfg.fps) as sync_mode:
        while True:
                if should_quit():
                    break

                pygame_clock.tick()
                snapshot, image_rgb, image_semseg = sync_mode.tick(timeout=1.0)

                waypoint = map.get_waypoint(ego_vehicle.get_location())
                waypoint_list = []
                for i in range(0, cfg.number_of_lanepoints):
                    waypoint_list.append(waypoint.next(i + cfg.meters_per_frame)[0])
                
                    # Convert and reshape image from Nx1 to shape(720, 1280, 3)
                image_semseg.convert(carla.ColorConverter.CityScapesPalette)
                image_semseg = reshape_image(image_semseg)
                
                lanes_list, x_lanes_list = lanemarkings.detect_lanemarkings(waypoint_list, image_semseg, camera_rgb)

                render_display(display, image_rgb, image_semseg, lanes_list, lanemarkings.colormap, cfg.render_lanes)

        for vehicle in world.get_actors().filter('*vehicle*'):
            vehicle.destroy()
        print("All vehicles destroyed")
        camera_rgb.destroy()
        camera_semseg.destroy()
        print("Cameras destroyed")

if __name__ == '__main__':
    client = carla.Client('localhost', 2000)
    world = client.get_world()

    main(client, world)
    pygame.quit()

    # for actor in world.get_actors():
    #     actor.destroy()





