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
    def __init__(self, run_name):
        self.display = pygame.display.set_mode((cfg.image_width, cfg.image_height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.pygame_clock = pygame.time.Clock()

        self.client = carla.Client('localhost', 2000)
        self.world = self.client.load_world(cfg.town)
        self.world.set_weather(cfg.weather)
        self.map = self.world.get_map()

        # Traffic manager
        self.tm = self.client.get_trafficmanager()
        self.tm.set_hybrid_physics_mode(True)
        self.tm.set_respawn_dormant_vehicles(True)

        blueprint_library = self.world.get_blueprint_library()
        self.spawn_points = self.world.get_map().get_spawn_points()

        # Spawn ego vehicle
        bp_ego_vehicle = random.choice(blueprint_library.filter('vehicle.ford.mustang'))
        bp_ego_vehicle.set_attribute('role_name', 'hero')
        self.ego_vehicle = self.world.spawn_actor(bp_ego_vehicle, random.choice(self.spawn_points))
        self.ego_vehicle.set_autopilot(True, self.tm.get_port())
        self.tm.update_vehicle_lights(self.ego_vehicle, True)
        # self.tm.keep_right_rule_percentage(self.ego_vehicle, 10)
        # self.tm.random_left_lanechange_percentage(self.ego_vehicle, 100)

        # Spawn rgb-cam and attach to vehicle
        bp_camera_rgb = blueprint_library.find('sensor.camera.rgb')
        bp_camera_rgb.set_attribute('image_size_x', f'{cfg.image_width}')
        bp_camera_rgb.set_attribute('image_size_y', f'{cfg.image_height}')
        bp_camera_rgb.set_attribute('fov', f'{cfg.fov}')
        camera_spawnpoint = carla.Transform(carla.Location(x=1.0, z=2.0), carla.Rotation(pitch=-18.5)) # camera 5
        self.camera_rgb = self.world.spawn_actor(bp_camera_rgb, camera_spawnpoint, attach_to=self.ego_vehicle)

        # Spawn semseg-cam and attach to vehicle
        bp_camera_semseg = blueprint_library.find('sensor.camera.semantic_segmentation')
        bp_camera_semseg.set_attribute('image_size_x', f'{cfg.image_width}')
        bp_camera_semseg.set_attribute('image_size_y', f'{cfg.image_height}')
        bp_camera_semseg.set_attribute('fov', f'{cfg.fov}')
        self.camera_semseg = self.world.spawn_actor(bp_camera_semseg, camera_spawnpoint, attach_to=self.ego_vehicle)

        # Spawn depth-cam and attach to vehicle
        bp_camera_depth = blueprint_library.find('sensor.camera.depth')
        bp_camera_depth.set_attribute('image_size_x', f'{cfg.image_width}')
        bp_camera_depth.set_attribute('image_size_y', f'{cfg.image_height}')
        bp_camera_depth.set_attribute('fov', f'{cfg.fov}')
        self.camera_depth = self.world.spawn_actor(bp_camera_depth, camera_spawnpoint, attach_to=self.ego_vehicle)

        # Spawn vehicles
        self.vehicle_blueprints = blueprint_library.filter('*vehicle*')
        self.vehicle_state = {self.ego_vehicle.id: self.get_vehicle_state(self.ego_vehicle)}
        self.respawn_interval = cfg.respawn * cfg.fps

        i = 0
        while i < cfg.num_vehicles:
            vehicle = self.world.try_spawn_actor(random.choice(self.vehicle_blueprints), random.choice(self.spawn_points))
            if vehicle is not None:
                vehicle.set_autopilot(True, self.tm.get_port())
                self.tm.update_vehicle_lights(vehicle, True)
                i += 1
            
        self.lanemarkings = LaneMarkings(self.client, self.world)

        self.tick_counter = 0
        self.colors = [[1,1,1], [2,2,2], [3,3,3], [4,4,4]]
        self.RGB_colors = [(0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 0, 255)]
        self.BGR_colors = [(0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 0)]

        # Create opencv window
        cv2.namedWindow("inst_background", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("inst_background", 640, 360)

        # Create save path
        if cfg.saving:
            if run_name:
                run_root = os.path.join(cfg.data_root, f"{cfg.town}_{run_name}")
            else:
                run_root = os.path.join(cfg.data_root, f"{cfg.town}")
            self.img_folder = os.path.join(run_root, "img")
            self.seg_folder = os.path.join(run_root, "seg")
            os.makedirs(run_root, exist_ok=False)
            os.makedirs(self.img_folder, exist_ok=True)
            os.makedirs(self.seg_folder, exist_ok=True)
        
            self.save_interval = cfg.save_freq * cfg.fps
            self.save_counter = 0
            self.skip_counter = 0
            self.skip_interval = cfg.skip_at_traffic_light_interval

            txt_file_path = os.path.join(run_root, "train_gt.txt")
            open(txt_file_path, "w").close()
            self.txt_fp = open(txt_file_path, "a", buffering=1)

    def get_vehicle_state(self, vehicle):
        t = vehicle.get_transform().location
        v = vehicle.get_velocity()
        return (round(t.x, 2), round(t.y, 2), round(t.z, 2),
                round(v.x, 2), round(v.y, 2), round(v.z, 2))


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


    def render_display(self, image_rgb, image_depth, lanes_list):
        """
        Renders the images captured from both cameras and shows it on the
        pygame display

        Args:
            image: numpy array. Shows the 3-channel numpy imagearray on the pygame display.
            image_semseg: numpy array. Shows the semantic segmentation image on the pygame display.
        """
        # Draw the display.
        image_rgb = self.reshape_image(image_rgb)
        image_depth = self.reshape_image(image_depth)
        self.draw_image(self.display, image_rgb)
        
        inst_background = np.zeros_like(image_rgb)
        inst_background_display = np.zeros_like(image_rgb)

        lane_exist = [0] * 4
        lane_cls = [0] * 4

        # Draw lanepoints of every lane on pygame window
        if(cfg.render_lanes):
            for i, color in enumerate(self.lanemarkings.colormap):
                lane = lanes_list[i]
                segments = [] # store all segments for a lane
                segment = [] # store all coordinates for a segment

                for x, y in lane:
                    if x == -2 or x == -1:
                        if segment: # avoid appending empty segment
                            segments.append(segment)
                            segment = []
                    else:
                        segment.append([x, y])
                        pygame.draw.circle(self.display, self.lanemarkings.colormap[color], (x, y), 3, 2)
                if segment: # append leftover segment
                    segments.append(segment) 

                if segments:
                    lane_exist[i] = 1 # mark lane as exist
                    max_seg = max(segments, key=len)

                    # # backproject lane in 2D pixel to 3D world to find lane type
                    mid_pixel = max_seg[len(max_seg) // 2] # (x, y)
                    lane_marking_type = self.lanemarkings.calculate_lane_marking_type_from_2Dlane(mid_pixel, i, image_depth, self.camera_depth)
                    # # self.world.debug.draw_point(w, size=0.2, color=carla.Color(255,255,0), life_time=2 * (1/cfg.fps), persistent_lines=False)    

                    lane_cls[i] = lane_marking_type

                    cv2.polylines(inst_background, np.int32([max_seg]), isClosed=False, color=self.colors[i], thickness=5)
                    cv2.polylines(inst_background_display, np.int32([max_seg]), isClosed=False, color=self.BGR_colors[i], thickness=5)
        
        pygame.display.flip()
        cv2.imshow("inst_background", inst_background_display)
        cv2.waitKey(1)

        if cfg.saving:
            if self.tick_counter % self.save_interval == 0:
                if self.ego_vehicle.is_at_traffic_light() or self.get_vehicle_state(self.ego_vehicle) == self.vehicle_state[self.ego_vehicle.id]:
                    if self.skip_counter % self.skip_interval != 0:
                        self.skip_counter += 1
                        print("save skipped at traffic")
                        return
                    else:
                        self.skip_counter += 1
                else:
                    self.skip_counter = 0 # reset skip counter

                    
                self.vehicle_state[self.ego_vehicle.id] = self.get_vehicle_state(self.ego_vehicle)

                # saving images
                image_rgb = Image.fromarray(image_rgb)
                inst_background = Image.fromarray(inst_background[:,:,0].astype(np.uint8), mode="L")
                img_path = os.path.join(self.img_folder, f"{self.save_counter}.png")
                seg_path = os.path.join(self.seg_folder, f"{self.save_counter}.png")
                image_rgb.save(img_path)
                inst_background.save(seg_path)

                # write to txt file
                s = [img_path, seg_path]
                s.extend([str(x) for x in lane_exist])
                s.extend([str(x) for x in lane_cls])
                s = " ".join(s) + "\n"
                self.txt_fp.write(s)

                print("saved:", self.save_counter)

                self.save_counter += 1

                if self.save_counter == cfg.save_num:
                    sys.exit()


    def run(self):
        with CarlaSyncMode(self.world, self.tm, self.camera_rgb, self.camera_semseg, self.camera_depth, fps=cfg.fps) as sync_mode:
            try:
                while True:
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

                    if self.tick_counter % self.respawn_interval == 0:
                        print("Checking vehicle states...")
                        vehicles = self.world.get_actors().filter("vehicle.*")
                        
                        for vehicle in vehicles:
                            # Skip ego vehicle
                            role_name = vehicle.attributes.get("role_name", "")
                            if role_name == "hero":
                                continue

                            current_state = self.get_vehicle_state(vehicle)
                            if vehicle.id in self.vehicle_state:
                                if current_state == self.vehicle_state[vehicle.id]:
                                    print(f"[STUCK] Vehicle {vehicle.id} hasn't moved. Respawning...")
                                    vehicle.destroy()

                                    # Respawn at a random valid point
                                    while True:
                                        new_vehicle = self.world.try_spawn_actor(random.choice(self.vehicle_blueprints), random.choice(self.spawn_points))
                                        if new_vehicle:
                                            new_vehicle.set_autopilot(True, self.tm.get_port())
                                            self.tm.update_vehicle_lights(new_vehicle, True)
                                            self.vehicle_state[new_vehicle.id] = self.get_vehicle_state(new_vehicle)
                                            break
                                else:
                                    self.vehicle_state[vehicle.id] = current_state
                            else:
                                self.vehicle_state[vehicle.id] = current_state
                
                    # clock tick
                    self.pygame_clock.tick()
                    snapshot, image_rgb, image_semseg, image_depth = sync_mode.tick(timeout=1.0)
                    self.tick_counter += 1

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

                    if cfg.draw3DLanes:
                        for waypoint in waypoint_list:
                            self.world.debug.draw_point(location=waypoint.transform.location, size=0.05, life_time=2 * (1/cfg.fps), persistent_lines=False)                    
                    
                    # Show lanes on pygame
                    self.render_display(image_rgb, image_depth, lanes_list)

            finally:
                # Destroy all actors after game ends
                for vehicle in self.world.get_actors().filter('*vehicle*'):
                    vehicle.destroy()
                print("All vehicles destroyed")
                self.camera_rgb.destroy()
                self.camera_semseg.destroy()
                self.camera_depth.destroy()
                print("Cameras destroyed")
                if cfg.saving:
                    self.txt_fp.close()
                    print("File saved")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", required=False, default=None)
    args = parser.parse_args()

    pygame.init()

    game = CarlaGame(args.run_name)
    game.run()

    pygame.quit()






