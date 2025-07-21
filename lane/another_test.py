import carla
import time

client = carla.Client('localhost', 2000)
world = client.load_world("Town04")
tm = client.get_trafficmanager()
print("sleeping...")
time.sleep(100)
