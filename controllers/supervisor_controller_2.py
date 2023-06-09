# Python imports
import random
import struct
import sys

sys.path.append('C:/Program Files/Webots/lib/controller/python38/')

print("Ok 1")
# Webots imports
from controller import Supervisor

# Global constants
PI = 3.141592653589793
RECEIVER_SAMPLING_PERIOD = 64	# In milliseconds

# /!\ Set the number of obstacle cars
NB_OBSTACLE_CARS = 3

# Clipping function
def value_clip(x, low, up):
	return low if x < low else up if x > up else x

# Angle normalization function (Webots angle values range between -pi and pi)
def angle_clip(x):
	a = x % (2*PI)
	return a if a < PI else a - 2*PI

# Random initial positions
# 	[[x_min, x_max], [y_min, y_max], rotation]
# All rotation values should correspond to the same driving direction on the track

# Train track
starting_positions = [
 	[[ 1.00,  3.50], [ 5.10,  5.30],  0.00],
 	[[ 4.65,  4.85], [ 0.30,  4.70], -PI/2],
 	[[ 3.15,  3.35], [ 1.70,  2.40],  PI/2],
 	[[ 0.15,  0.35], [ 0.20,  1.60], -PI/2],
 	[[-1.80, -1.60], [ 0.00,  2.20],  PI/2]
]

# Test track
"""
starting_positions = [
	[[ 0.60,  4.20], [ 5.40,  5.60],  0.00],
	[[ 5.40,  5.60], [-4.30,  4.30], -PI/2],
	[[ 1.40,  1.60], [-5.00, -4.30],  PI/2],
	[[ 3.40,  3.60], [-2.85, -1.45],  PI/2],
	[[ 3.40,  3.60], [ 2.20,  2.50],  PI/2],
	[[-0.60, -0.40], [ 0.30,  0.90], -PI/2],
	[[-3.20, -2.00], [-4.60, -4.40],  PI],
	[[-2.60, -2.40], [-0.80,  2.30],  PI/2],
]
"""

# Initialisation
supervisor = Supervisor()
basicTimeStep = int(supervisor.getBasicTimeStep())

# Receiver
receiver = supervisor.getDevice("receiver")
receiver.enable(RECEIVER_SAMPLING_PERIOD)

# Get all car nodes
tt_02 = supervisor.getFromDef("TT02")
tt_02_translation = tt_02.getField("translation")
tt_02_rotation = tt_02.getField("rotation")
obstacle_car_nodes = [supervisor.getFromDef(f"obstacle_car_{i}") for i in range(NB_OBSTACLE_CARS)]
obstacle_car_translation_fields = [obstacle_car_nodes[i].getField("translation") for i in range(NB_OBSTACLE_CARS)]
obstacle_car_rotation_fields = [obstacle_car_nodes[i].getField("rotation") for i in range(NB_OBSTACLE_CARS)]

# Main loop
while supervisor.step(basicTimeStep) != -1:

	# If reset signal : replace the cars
	if receiver.getQueueLength() > 0:
		# Get the data off the queue
		data = receiver.getData()
		number = struct.unpack('i', data)[0]
		receiver.nextPacket()

		# Empty queue
		while receiver.getQueueLength() > 0:
			data = receiver.getData()
			number = struct.unpack('i', data)[0]
			receiver.nextPacket()

		# Choose driving direction
		direction = random.choice([0, 1])
		# Select random starting points
		coordinates_idx = random.sample(range(len(starting_positions)), 1+NB_OBSTACLE_CARS)

		# Replace TT-02
		coords = starting_positions[coordinates_idx[0]]
		start_x = random.uniform(coords[0][0], coords[0][1])
		start_y = random.uniform(coords[1][0], coords[1][1])
		start_z = 0.04
		tt_02_translation.setSFVec3f([start_x, start_y, start_z])
		# Rotate TT-02
		angle = coords[2]
		start_angle = random.uniform(angle - PI/12, angle + PI/12)
		if direction:
			start_angle = start_angle + PI
		start_rot = [0, 0, 1, angle_clip(start_angle)]
		tt_02_rotation.setSFRotation(start_rot)

		# Replace obstacle cars
		for i in range(NB_OBSTACLE_CARS):
			coords = starting_positions[coordinates_idx[i + 1]]
			start_x = random.uniform(coords[0][0], coords[0][1])
			start_y = random.uniform(coords[1][0], coords[1][1])
			start_z = 0.04
			obstacle_car_translation_fields[i].setSFVec3f([start_x, start_y, start_z])
			# Rotate obstacle cars
			angle = coords[2]
			start_angle = random.uniform(angle - PI/12, angle + PI/12)
			if direction:
				start_angle = start_angle + PI
			start_rot = [0, 0, 1, angle_clip(start_angle)]
			obstacle_car_rotation_fields[i].setSFRotation(start_rot)
