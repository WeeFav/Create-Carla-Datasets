import carla

# Camera
fps = 20
image_width = 1280
image_height = 720
fov = 90

# Lanes
meters_per_frame = 1.0
number_of_lanepoints = 80
junctionMode = True
render_lanes = True
draw3DLanes = False

row_anchor_start = 160
h_samples = []
for y in range(row_anchor_start, image_height, 10):
	h_samples.append(y)

# World
town = 'Town10HD_Opt'
num_vehicles = 50
# [ClearNoon, CloudyNoon, WetNoon, WetCloudyNoon, SoftRainNoon, MidRainyNoon, HardRainNoon, ClearSunset, CloudySunset, WetSunset, WetCloudySunset, SoftRainSunset, MidRainSunset, HardRainSunset]
weather = carla.WeatherParameters.CloudySunset

# Mode
auto_run = True
saving = True
data_root = "C:\\Users\\marvi\\Datasets\\Lane\\CarlaLane"
save_freq = 4 # in seconds
skip_at_traffic_light_interval = 5 # number of saved frames
respawn = 60 # in seconds