from vehicle import Driver
import numpy as np

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

with open("../controller_pre_trainer/outputs.txt") as file:
    print("Initializing")
    driver = Driver()
    basicTimeStep = int(driver.getBasicTimeStep())
    sensorTimeStep = 4 * basicTimeStep

    index = 0
    data = file.readline().split("][")

    while driver.step() != -1:
        driver.step()
        instruction = data[index].strip().replace("[", "").replace("]", "")
        print(instruction)
        speed = float(instruction.split()[0].strip())
        steering = float(instruction.split()[1].strip())

        index += 1

        driver.setCruisingSpeed(speed)
        driver.setSteeringAngle(steering)