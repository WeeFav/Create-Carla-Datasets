fps = 20
image_width = 1280
image_height = 720
fov = 90

meters_per_frame = 1.0
number_of_lanepoints = 80
junctionMode = True
render_lanes = True

row_anchor_start = 160

h_samples = []
for y in range(row_anchor_start, image_height, 10):
	h_samples.append(y)