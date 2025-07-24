import carla
import math
import numpy as np
from collections import deque
import config as cfg

# ==============================================================================
# -- Lane Markings ------------------------------------------------------------
# ==============================================================================

class LaneMarkings():
    """
    Helper class to detect and draw lanemarkings in carla.
    """
    def __init__(self, client, world):
        self.colormap = {'green': (0, 255, 0),
                         'red': (255, 0, 0),
                         'yellow': (255, 255, 0),
                         'blue': (0, 0, 255)}
        
        self.colormap_carla = {'green': carla.Color(0, 255, 0),
                               'red': carla.Color(255, 0, 0),
                               'yellow': carla.Color(255, 255, 0),
                               'blue': carla.Color(0, 0, 255)}

        # Intrinsic camera matrix needed to convert 3D-world coordinates to 2D-imagepoints
        self.f = cfg.image_width / (2 * math.tan(cfg.fov * math.pi / 360))
        self.c_x = cfg.image_width/2
        self.c_y = cfg.image_height/2
        
        self.cameraMatrix  = np.float32([[self.f, 0, self.c_x],
                                         [0, self.f, self.c_y],
                                         [0, 0, 1]])
        
        self.lanes = [
            deque(maxlen=cfg.number_of_lanepoints), 
            deque(maxlen=cfg.number_of_lanepoints), 
            deque(maxlen=cfg.number_of_lanepoints), 
            deque(maxlen=cfg.number_of_lanepoints)
        ]

        self.client = client
        self.world = world
    

    def draw_points(self, client, point):
        if point is not None:
            client.get_world().debug.draw_point(point + carla.Location(z=0.05), size=0.05, life_time=2 * (1/cfg.fps), persistent_lines=False)    
    
    
    def draw_lanes(self, client, point0, point1, color):
        if(point0 and point1):
            client.get_world().debug.draw_line(point0 + carla.Location(z=0.05), point1 + carla.Location(z=0.05), thickness=0.05, 
                color=color, life_time=cfg.number_of_lanepoints/cfg.fps, persistent_lines=False)
        
    
    def calculate3DLanepoints(self, lanepoint):
        """
        Calculates the 3-dimensional position of the lane from the lanepoint from the informations given. 
        The information analyzed is the lanemarking type, the lane type and for the calculation the 
        lanepoint (the middle of the actual lane), the lane width and the driving direction of the lanepoint. 
        In addition to the Lanemarking position, the function does also calculate the position of respectively 
        adjacent lanes. The actual implementation includes contra flow lanes.
        
        Args:
            client: carla.Client. The client is used to draw 3-dimensional lanepoints.
            lanepoint: carla.Waypoint. The lanepoint, of wich the lanemarkings should be calculated.

        Returns:
            lanes: list of 4 lanelists. 3D-positions of every lanemarking of the given lanepoints actual lane, and corresponding neighbour lanes if they exist.
        """
        orientationVec = lanepoint.transform.get_forward_vector()
        
        length = math.sqrt(orientationVec.y*orientationVec.y+orientationVec.x*orientationVec.x)
        abVec = carla.Location(orientationVec.y,-orientationVec.x,0) / length * 0.5* lanepoint.lane_width
        right_lanemarking = lanepoint.transform.location - abVec 
        left_lanemarking = lanepoint.transform.location + abVec

        if(lanepoint.left_lane_marking.type == carla.LaneMarkingType.NONE):
            left_lanemarking = None
        self.lanes[1].append(left_lanemarking)

        if(lanepoint.right_lane_marking.type == carla.LaneMarkingType.NONE):    
            right_lanemarking = None
        self.lanes[2].append(right_lanemarking)

        # Calculate left lanes
        if(lanepoint.get_left_lane() and lanepoint.get_left_lane().left_lane_marking.type != carla.LaneMarkingType.NONE):
            outer_left_lanemarking  = lanepoint.transform.location + 3 * abVec
        else:
            outer_left_lanemarking = None
        self.lanes[0].append(outer_left_lanemarking)

        # Calculate right lanes
        if(lanepoint.get_right_lane() and lanepoint.get_right_lane().right_lane_marking.type != carla.LaneMarkingType.NONE):
            outer_right_lanemarking = lanepoint.transform.location - 3 * abVec
        else:
            outer_right_lanemarking = None
        self.lanes[3].append(outer_right_lanemarking)
        
        if cfg.draw3DLanes:
            self.draw_points(self.client, left_lanemarking)
            self.draw_points(self.client, right_lanemarking) 
            self.draw_points(self.client, outer_left_lanemarking)
            self.draw_points(self.client, outer_right_lanemarking)
        
        return self.lanes
    
    def my_calculate3DLanepoints(self, waypoint):
        orientationVec = waypoint.transform.get_forward_vector()
        
        length = math.sqrt(orientationVec.y*orientationVec.y+orientationVec.x*orientationVec.x)
        abVec = carla.Location(orientationVec.y,-orientationVec.x,0) / length * 0.5* waypoint.lane_width

        if not waypoint.is_junction:
            left_lanemarking = waypoint.transform.location + abVec
            right_lanemarking = waypoint.transform.location - abVec 
        else:
            left_lanemarking = None
            right_lanemarking = None
        self.lanes[1].append(left_lanemarking)
        self.lanes[2].append(right_lanemarking)


        # Calculate left lanes
        left_waypoint = waypoint.get_left_lane()
        if (left_waypoint and left_waypoint.lane_type == carla.LaneType.Driving and not left_waypoint.is_junction):
            outer_left_lanemarking = waypoint.transform.location + 3 * abVec
        else:
            outer_left_lanemarking = None
        self.lanes[0].append(outer_left_lanemarking)


        # Calculate right lanes
        right_waypoint = waypoint.get_right_lane()
        if (right_waypoint and right_waypoint.lane_type == carla.LaneType.Driving and not right_waypoint.is_junction):
            outer_right_lanemarking = waypoint.transform.location - 3 * abVec
        else:
            outer_right_lanemarking = None
        self.lanes[3].append(outer_right_lanemarking)


        if cfg.draw3DLanes:
            self.draw_points(self.client, left_lanemarking)
            self.draw_points(self.client, right_lanemarking) 
            self.draw_points(self.client, outer_left_lanemarking)
            self.draw_points(self.client, outer_right_lanemarking)
        
        return self.lanes


    def calculate2DLanepoints(self, camera_rgb, lane_list):
        """
        Transforms the 3D-lanepoint coordinates to 2D-imagepoint coordinates. If there's a huge hole in the list (None values),
        we need to split the list into two flat_lane_lists to make sure, the lanepoints are calculated and shown properly.
        
        Args:
            camera_rgb: carla.Actor. Get the camera_rgb actor to calculate the extrinsic matrix with the help of inverse matrix.
            lane_list: list. List of a lane, which contains its 3D-lanepoint coordinates x, y and z.

        Returns:
            List of 2D-points, where the elements of the list are tuples. Each tuple contains an x and y value.
        """
        lane_list = list(filter(lambda x: x!= None, lane_list))

        flat_lane_list = []  
        flat_lane_list_a = []

        if lane_list:    
            last_lanepoint = lane_list[0]
            
        for lanepoint in lane_list:
            # Draw outer lanes not on junction
            # calculate distance between current point and previous point
            distance = math.sqrt(math.pow(lanepoint.x-last_lanepoint.x ,2)+math.pow(lanepoint.y-last_lanepoint.y ,2)+math.pow(lanepoint.z-last_lanepoint.z ,2))
        
            # Check of there's a hole in the list
            if distance > cfg.meters_per_frame * 3: # if distance is too large, there is a gap
                flat_lane_list.append(flat_lane_list_a.copy())
                flat_lane_list_a = []
                last_lanepoint = lanepoint
                continue
                            
            last_lanepoint = lanepoint
            flat_lane_list_a.append([lanepoint.x, lanepoint.y, lanepoint.z, 1.0])

        flat_lane_list.append(flat_lane_list_a.copy()) # append the last segment
        
        flat_lane_list_2D = []
        for segment in flat_lane_list:
            if segment:
                world_points = np.float32(segment).T
                
                # This (4, 4) matrix transforms the points from world to sensor coordinates.
                world_2_camera = np.array(camera_rgb.get_transform().get_inverse_matrix())
                
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
                x_coord = points_2d[:, 0].astype(int)
                y_coord = points_2d[:, 1].astype(int)
            else:
                x_coord = np.array([])
                y_coord = np.array([])

            flat_lane_list_2D.append((x_coord.copy(), y_coord.copy()))

        x_coord = []
        y_coord = []
        for seg_2D in flat_lane_list_2D[:-1]: # for all segments except last one
            seg_2D_x = seg_2D[0]
            seg_2D_y = seg_2D[1]

            seg_2D_x = np.concatenate([seg_2D_x, [-1]]) # append -1 to the end of that segment
            seg_2D_y = np.concatenate([seg_2D_y, [-1]]) # append -1 to the end of that segment

            x_coord.append(seg_2D_x)
            y_coord.append(seg_2D_y)

        x_coord.append(flat_lane_list_2D[-1][0]) # add the last segment
        y_coord.append(flat_lane_list_2D[-1][1]) # add the last segment

        x_coord = np.concatenate(x_coord)
        y_coord = np.concatenate(y_coord)

        # x_coord = [x1, x2, ..., -1, x100, x101, ...]
        # x1, x2, are from new_x_coord, representing first segment (closer to bottom of screen)
        # x100, x101 are from old_x_coord, representing second segment (closer to top of screen)

        return list(zip(x_coord, y_coord))


    def calculateYintersections(self, lane_list):
        """
        Transforms the 2D-image coordinates to the correct input format for the deep learning model, where only the y-values in steps of 10 are needed.
        This is done by the intersection of the line of the two points enclosing the y-value and the line f(x) = y. 
        For the lower half (0.6) of the image, if there are no points enclosing the y-value, the intersection is calculate by the first points existing.
        This may leads to inaccurate results at junctions, but completes the lines until the bottom of the image. 
        Since junctions are excluded from trainingsdata the incorrect results at junctions are as well.
        
        Args:
            lane_list: list. List of a lane, which contains its 2D-image-coordinates x, y.

        Returns:
            List of 2D-points in the correct format for the deep learning algorithm, where the elements of the list 
            are tuples. Each tuple contains an x and y value.
        """
        x_coord = []
        gap = False
        
        if(len(lane_list) > 2):
            last_point = lane_list[0]
            for xy_val in lane_list:
                if last_point == xy_val:
                    continue
                if xy_val[0]==-1:
                    gap = True
                    continue
                if last_point[0]==-1:
                    gap = True
                    last_point=[0.5* cfg.image_width, cfg.image_height-1]
                for y_value in reversed(cfg.h_samples):
                    if gap and (last_point[1] >= y_value and xy_val[1] < y_value):
                        x_coord.append(-2)
                    elif (last_point == lane_list[0] and last_point[1] < y_value) and last_point[1] < cfg.image_height - 0.4 * cfg.image_height :
                        x_coord.append(-2)
                    elif (last_point[1] >= y_value and xy_val[1] < y_value) or (last_point == lane_list[0] and last_point[1] < y_value) and last_point[1] >= cfg.image_height - 0.4 * cfg.image_height:
                        if last_point[1]-xy_val[1] == 0:
                            intersection = last_point[1]
                        else:
                            intersection = xy_val[0] + ((y_value-xy_val[1])*(last_point[0]-xy_val[0]))/(last_point[1]-xy_val[1])
                        if intersection >= cfg.image_width or intersection < 0:
                            x_coord.append(-2)
                        else:
                            x_coord.append(int(intersection))
                gap = False
                last_point = xy_val

            while len(x_coord) < len(cfg.h_samples):
                x_coord.append(-2)
            return list(zip(reversed(x_coord), cfg.h_samples))
        else:
            for i in cfg.h_samples:
                x_coord.append(-2)
            return list(zip(x_coord, cfg.h_samples))


    def filter2DLanepoints(self, lane_list, image):
        """
        Remove all calculated 2D-lanepoints from the lane_list, which are e.g. on a house or wall, with the help of the sematic segmentation camera.
  
        Args:
            lane_list: list. List of a lane, which contains its lanepoints. Needed to check, if a lanepoint is located on a semantic tag like road or roadline.
            image: numpy array. Semantic segmentation image providing specific colorlabels to identify, if road or roadline.

        Returns:
            filtered_lane_list: list. Every point, which is located on a road or roadline, but not on a building or wall.
        """
        filtered_lane_list = []
        for lanepoint in lane_list:
            x = lanepoint[0]
            y = lanepoint[1]
            if(np.any(image[y][x] == (128, 64, 128), axis=-1) or   # Road
               np.any(image[y][x] == (157, 234, 50), axis=-1) or   # Roadline
               np.any(image[y][x] == (244, 35, 232), axis=-1) or   # Sidewalk
               np.any(image[y][x] == (220, 220, 0), axis=-1) or    # Traffic sign
               np.any(image[y][x] == (0, 0, 142), axis=-1)):       # Vehicle
                filtered_lane_list.append(lanepoint)
            else:
                filtered_lane_list.append((-2, y))
                
        return filtered_lane_list


    def format2DLanepoints(self, lane_list):
        """
        Args: 
            lane_list: list. List of a lane, which contains its lanepoints. 

        Returns:
            x_lane_list: list of x-coordinates. Extracts all the x values from the lane_list.
            Formats the lane_list as followed: [(x0,y0),(x1,y1),...,(xn,yn)] to [x0, x1, ..., xn]
        """
        x_lane_list = []
        for lanepoint in lane_list:
            x_lane_list.append(lanepoint[0])
        
        return x_lane_list


    def detect_lanemarkings(self, waypoint_list, image_semseg, camera_rgb):
        """
        Calculate and show all lanes on the road.

        Args:
            new_waypoint: carla.Waypoint. Calculate all the lanemarkings based on the new_waypoint, which is the last element from the waypoint_list.
            image_semseg: numpy array. Filter lanepoints with semantic segmentation camera. 
        """
        lanes_list = []      # filtered 2D-Points
        x_lanes_list = []    # only x values of lanes

        for lanepoint in waypoint_list:
            lanes_3Dcoords = self.calculate3DLanepoints(lanepoint)
        
        for lane_3Dcoords in lanes_3Dcoords:
            lane = self.calculate2DLanepoints(camera_rgb, lane_3Dcoords)
            lane = self.calculateYintersections(lane)
            lane = self.filter2DLanepoints(lane, image_semseg)
            lanes_list.append(lane)

            x_lanes_list.append(self.format2DLanepoints(lane))
        
        return lanes_list, x_lanes_list
    

    def calculate_lane_marking_type_from_2Dlane(self, point, lane_num, image_depth, camera_depth):
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
        point_world = camera_depth.get_transform().transform(point_cam)
        lanepoint = carla.Location(x=point_world.x, y=point_world.y, z=point_world.z)
        lane_marking_type, side = self.get_lane_marking_type(lanepoint)

        if lane_marking_type == carla.LaneMarkingType.Broken:
            return 1
        elif lane_marking_type == carla.LaneMarkingType.Solid or lane_marking_type == carla.LaneMarkingType.SolidSolid:
            return 2
        else:
            return 3

    def get_lane_marking_type(self, location):        
        # Step 1: Project to closest waypoint
        waypoint = self.world.get_map().get_waypoint(
            location,
            project_to_road=True,
            lane_type=carla.LaneType.Any
        )
        # self.world.debug.draw_point(waypoint.transform.location, size=0.2, color=carla.Color(255,255,0), life_time=2 * (1/cfg.fps), persistent_lines=False)

        # Step 2: Vector from waypoint (lane center) to original location
        dx = location.x - waypoint.transform.location.x
        dy = location.y - waypoint.transform.location.y

        # Step 4: Compute perpendicular (normal vector to the right of lane)
        right = waypoint.transform.get_right_vector() 

        # Step 5: Project vector from center to location onto the right vector
        dot = dx * right.x + dy * right.y

        # Step 6: Determine side and return the correct marking
        if dot >= 0:
            # Location is to the right of lane center
            marking = waypoint.right_lane_marking
            side = "right"
        else:
            # Location is to the left
            marking = waypoint.left_lane_marking
            side = "left"

        return marking.type, side