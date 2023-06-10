from vehicle import Driver
from controller import Lidar
import numpy

driver = Driver()
basicTimeStep = int(driver.getBasicTimeStep())
sensorTimeStep = 4 * basicTimeStep

lidar = Lidar("lidar")
lidar.enable(sensorTimeStep)
lidar.enablePointCloud()

max_angle = 16
edge_weight = 3.0

def to_radians_clamped(angle):
    if angle > max_angle:
        angle = max_angle
    elif angle < -max_angle:
        angle = -max_angle
    return angle * 3.14 / 180

def determine_largest_gap(data):
    largest_gap_width = 0
    largest_gap_heading = 0

    for i in range(len(data)):
        gap_width = 0
        for j in range(len(data) - i):
            if data[i + j] > 3:
                gap_width += 1
            else:
                break

        if gap_width > largest_gap_width:
            largest_gap_width = gap_width
            largest_gap_heading = i + (gap_width / 2)

    if largest_gap_width > 5:
        return largest_gap_heading - 100

    return 0


def rotate_heading_by_edges(data, heading):
    left_edge = data[50]
    right_edge = data[149]

    if left_edge >= 5:
        return heading - 5
    elif right_edge >= 5:
        return heading + 5
    else:
        edge_difference = right_edge - left_edge
        return heading + edge_difference * edge_weight


while driver.step() != -1:
    while True:
        driver.step()
        lidar_data = lidar.getRangeImage()

        gap_heading = determine_largest_gap(lidar_data)
        gap_heading = rotate_heading_by_edges(lidar_data, gap_heading)

        driver.setCruisingSpeed(4)
        driver.setSteeringAngle(to_radians_clamped(gap_heading))
