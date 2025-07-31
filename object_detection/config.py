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

# World
town = 'Town10HD_Opt'
num_vehicles = 50
# [ClearNoon, CloudyNoon, WetNoon, WetCloudyNoon, SoftRainNoon, MidRainyNoon, HardRainNoon, ClearSunset, CloudySunset, WetSunset, WetCloudySunset, SoftRainSunset, MidRainSunset, HardRainSunset]
weather = carla.WeatherParameters.ClearNoon # clear_noon
# weather = carla.WeatherParameters.HardRainSunset # hard_rain_sunset
# weather = carla.WeatherParameters( # night
# 	sun_altitude_angle=-90,
# )
# weather = carla.WeatherParameters( # fog_sunrise
# 	cloudiness=40,
# 	sun_altitude_angle=8,
# 	fog_density=10,
# )

# Mode
auto_run = True
saving = False
data_root = "C:\\Users\\marvi\\Datasets\\Lane\\CarlaLane"
save_freq = 3 # in seconds
skip_at_traffic_light_interval = 5 # number of saved frames
respawn = 45 # in seconds
save_num = 250