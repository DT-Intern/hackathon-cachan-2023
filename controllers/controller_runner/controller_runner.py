from vehicle import Driver
from controller import Lidar
from keras.models import load_model
import numpy as np
import math

model_path = "../controller_pre_trainer/pre_trainer_model.h5"
model = load_model(model_path)

driver = Driver()
basicTimeStep = int(driver.getBasicTimeStep())
sensorTimeStep = 4 * basicTimeStep

lidar = Lidar("lidar")
lidar.enable(sensorTimeStep)
lidar.enablePointCloud()

def set_speed_m_s(speed_m_s):
    speed = speed_m_s * 3.6
    if speed > 28:
        speed = 28
    if speed < 0:
        speed = 0
    driver.setCruisingSpeed(speed)

def set_direction_degree(angle_degree):
    if angle_degree > 16:
        angle_degree = 16
    elif angle_degree < -16:
        angle_degree = -16
    angle = angle_degree * 3.14 / 180
    driver.setSteeringAngle(angle)

while driver.step() != -1:
    lidar_data = lidar.getRangeImage()
    processed_data = np.array(lidar_data).reshape(1, 200)

    action = model.predict(processed_data)[0]
    speed, steering = action

    if math.isnan(speed) or math.isnan(steering):
        print("Nan speed or steering")
        continue

    if steering > 0.314:
        steering = 0.3
    elif steering < -0.314:
        steering = -0.3

    print(speed)
    print(steering)

    # set_speed_m_s(speed)
    # set_direction_degree(steering)

    driver.setCruisingSpeed(float(speed))
    driver.setSteeringAngle(float(steering))