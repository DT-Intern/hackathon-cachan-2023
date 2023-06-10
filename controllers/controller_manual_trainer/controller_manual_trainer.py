from vehicle import Driver
from controller import Lidar
import numpy as np

driver = Driver()

basicTimeStep = int(driver.getBasicTimeStep())
sensorTimeStep = 4 * basicTimeStep

lidar = Lidar("lidar")
lidar.enable(sensorTimeStep)
lidar.enablePointCloud()

keyboard = driver.getKeyboard()
keyboard.enable(sensorTimeStep)

speed = 0
max_speed = 28

angle = 0
max_angle = 16

def set_speed(speed_m_s):
    speed = speed_m_s * 3.6
    if speed > max_speed:
        speed = max_speed
    elif speed < 0:
        speed = 0
    driver.setCruisingSpeed(speed)

def set_angle(angle_degree):
    if angle_degree > max_angle:
        angle_degree = max_angle
    elif angle_degree < -max_angle:
        angle_degree = -max_angle
    angle = angle_degree * 3.14 / 180
    driver.setSteeringAngle(angle)

running = True
training_data = []
target_data = []

def to_internal_speed(raw):
    return raw * 3.6

def to_internal_angle(raw):
    return raw * 3.14 / 180

max_speed = 28
min_speed = -28
current_speed = 0

max_angle = 16
min_angle = -16
current_angle = 0

while driver.step() != -1:
    while running:
        driver.step()
        lidar_data = lidar.getRangeImage()

        current_key = keyboard.getKey()
        if current_key == ord("W"):
            # Increase speed
            current_speed = 3

            if current_speed > max_speed:
                current_speed = max_speed
        elif current_key == ord("S"):
            # Decrease speed
            current_speed = 0

            if current_speed < min_speed:
                current_speed = min_speed
        elif current_key == ord("A"):
            # Decrease turn radius
            # current_angle -= 4
            current_angle = -3.14

            if current_angle < min_angle:
                current_angle = min_angle
        elif current_key == ord("D"):
            # Increase turn radius
            # current_angle += 4
            current_angle = 3.14

            if current_angle > max_angle:
                current_angle = max_angle

        data_set_speed = current_speed / max_speed
        data_set_angle = current_angle / max_angle

        driver.setCruisingSpeed(to_internal_speed(current_speed))
        driver.setSteeringAngle(to_internal_angle(current_angle))

        # training_data.append(lidar_data)
        # target_data.append([data_set_speed, data_set_angle])
        # print("Collecting data", len(training_data))

        # if len(training_data) >= 1400:
            # training_data = np.array(training_data)
            # target_data = np.array(target_data)

            # print("Saving model")