import carla

# Camera
fps = 10
image_width = 1280
image_height = 720
fov = 90


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
saving = True
data_root = "C:\\Users\\marvi\\Datasets\\Object\\CarlaKitti"
save_freq = 3 # in seconds
skip_at_traffic_light_interval = 5 # number of saved frames
respawn = 45 # in seconds
save_num = 250
exclude_large_vehicle = False