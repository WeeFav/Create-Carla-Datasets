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
    def __init__(self, client):
        self.colormap = {'green': (0, 255, 0),
                         'red': (255, 0, 0),
                         'yellow': (255, 255, 0),
                         'blue': (0, 0, 255)}
        
        self.colormap_carla = {'green': carla.Color(0, 255, 0),
                               'red': carla.Color(255, 0, 0),
                               'yellow': carla.Color(255, 255, 0),
                               'blue': carla.Color(0, 0, 255)}

        # Intrinsic camera matrix needed to convert 3D-world coordinates to 2D-imagepoints
        f = cfg.image_width / (2 * math.tan(cfg.fov * math.pi / 360))
        c_x = cfg.image_width/2
        c_y = cfg.image_height/2
        
        self.cameraMatrix  = np.float32([[f, 0, c_x],
                                         [0, f, c_y],
                                         [0, 0, 1]])
        
        self.lanes = [
            deque(maxlen=cfg.number_of_lanepoints), 
            deque(maxlen=cfg.number_of_lanepoints), 
            deque(maxlen=cfg.number_of_lanepoints), 
            deque(maxlen=cfg.number_of_lanepoints)
        ]

        self.client = client
    

    def draw_points(self, client, point):
        client.get_world().debug.draw_point(point + carla.Location(z=0.05), size=0.05, life_time=cfg.number_of_lanepoints/self.fps, persistent_lines=False)    
    
    
    def draw_lanes(self, client, point0, point1, color):
        if(point0 and point1):
            client.get_world().debug.draw_line(point0 + carla.Location(z=0.05), point1 + carla.Location(z=0.05), thickness=0.05, 
                color=color, life_time=cfg.number_of_lanepoints/self.fps, persistent_lines=False)
        
    
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
        
        if(cfg.junctionMode):
            self.lanes[0].append(left_lanemarking) if(lanepoint.left_lane_marking.type != carla.LaneMarkingType.NONE) else self.lanes[0].append(None)
            self.lanes[1].append(right_lanemarking) if(lanepoint.right_lane_marking.type != carla.LaneMarkingType.NONE) else self.lanes[1].append(None)  
        
            # Calculate remaining outer lanes (left and right).
            if(lanepoint.get_left_lane() and lanepoint.get_left_lane().left_lane_marking.type != carla.LaneMarkingType.NONE):
                outer_left_lanemarking  = lanepoint.transform.location + 3 * abVec
                self.lanes[2].append(outer_left_lanemarking)
                #draw_points(self.client, outer_left_lanemarking)
            else:
                self.lanes[2].append(None)
    
            if(lanepoint.get_right_lane() and lanepoint.get_right_lane().right_lane_marking.type != carla.LaneMarkingType.NONE):
                outer_right_lanemarking = lanepoint.transform.location - 3 * abVec
                self.lanes[3].append(outer_right_lanemarking)
                #draw_points(self.client, outer_right_lanemarking)
            else:
                self.lanes[3].append(None)
            
            #draw_points(self.client, left_lanemarking)
            #draw_points(self.client, right_lanemarking)
        else:
            self.lanes[0].append(left_lanemarking) if left_lanemarking else self.lanes[0].append(None)
            self.lanes[1].append(right_lanemarking) if right_lanemarking else self.lanes[1].append(None)
        
            # Calculate remaining outer lanes (left and right).
            if(lanepoint.get_left_lane() and lanepoint.get_left_lane().lane_type == carla.LaneType.Driving):
                outer_left_lanemarking  = lanepoint.transform.location + 3 * abVec
                self.lanes[2].append(outer_left_lanemarking)
                #draw_points(client, outer_left_lanemarking)
            else:
                self.lanes[2].append(None)
                
            if(lanepoint.get_right_lane() and lanepoint.get_right_lane().lane_type == carla.LaneType.Driving):
                outer_right_lanemarking = lanepoint.transform.location - 3 * abVec
                self.lanes[3].append(outer_right_lanemarking)
                #draw_points(client, outer_right_lanemarking)
            else:
                self.lanes[3].append(None)
            
            #draw_points(client, left_lanemarking)
            #draw_points(client, right_lanemarking)
        
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
        flat_lane_list_a = []
        flat_lane_list_b = []
        lane_list = list(filter(lambda x: x!= None, lane_list))
        
        if lane_list:    
            last_lanepoint = lane_list[0]
            
        for lanepoint in lane_list:
            if(lanepoint and last_lanepoint):
                # Draw outer lanes not on junction
                distance = math.sqrt(math.pow(lanepoint.x-last_lanepoint.x ,2)+math.pow(lanepoint.y-last_lanepoint.y ,2)+math.pow(lanepoint.z-last_lanepoint.z ,2))
            
                # Check of there's a hole in the list
                if distance > cfg.meters_per_frame * 3:
                    flat_lane_list_b = flat_lane_list_a
                    flat_lane_list_a = []
                    last_lanepoint = lanepoint
                    continue
                
                last_lanepoint = lanepoint
                flat_lane_list_a.append([lanepoint.x, lanepoint.y, lanepoint.z, 1.0])

            else:
                # Just append a "Null" value
                flat_lane_list_a.append([None, None, None, None])
        
        if flat_lane_list_a:
            world_points = np.float32(flat_lane_list_a).T
            
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
            x_coord = []
            y_coord = []

        if flat_lane_list_b:
            world_points = np.float32(flat_lane_list_b).T
            
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
            
            old_x_coord = np.insert(x_coord, 0, -1)
            old_y_coord = np.insert(y_coord, 0, -1)
            
            # Extract the screen coords (xy) as integers.
            new_x_coord = points_2d[:, 0].astype(int)
            new_y_coord = points_2d[:, 1].astype(int)

            x_coord = np.concatenate((new_x_coord, old_x_coord), axis=None)
            y_coord = np.concatenate((new_y_coord, old_y_coord), axis=None)

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