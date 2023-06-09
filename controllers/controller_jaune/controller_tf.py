from keras.models import Sequential
from keras.layers import Dense
from vehicle import Driver
from controller import Lidar
import numpy as np

model = Sequential()
model.add(Dense(64, input_shape=(200,), activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(2))

model.compile(loss="mean_squared_error", optimizer="adam")

driver = Driver()
emitter = driver.getDevice("emitter")

basicTimeStep = int(driver.getBasicTimeStep())
sensorTimeStep = 4 * basicTimeStep

lidar = Lidar("lidar")
lidar.enable(sensorTimeStep)
lidar.enablePointCloud()

distance_traveled = 0

def collision_occurred(lidar_data, threshold_distance):
    for distance in lidar_data:
        if distance < threshold_distance:
            return True
    return False


def calculate_reward(speed, lidar_data):
    # Define reward weights
    speed_weight = 0.9
    progress_reward = 0.4
    collision_penalty = -10.0

    # Check for collision
    if collision_occurred(lidar_data, 0.1):
        return collision_penalty

    # Calculate progress reward based on distance traveled
    progress = distance_traveled / 1000
    progress_reward = progress * progress_reward

    # Calculate the overall reward
    reward = (speed_weight * speed) + progress_reward

    return reward

def set_speed_m_s(speed_m_s):
    speed = speed_m_s * 3.6
    if speed > 28:
        speed = 28
    if speed < 0:
        speed = 0
    driver.setCruisingSpeed(speed)
    return speed


def set_direction_degree(angle_degree):
    if angle_degree > 16:
        angle_degree = 16
    elif angle_degree < -16:
        angle_degree = -16
    angle = angle_degree * 3.14 / 180
    driver.setSteeringAngle(angle)
    return angle

print("Initialization complete")

while driver.step() == -1:
    pass

print("Starting learning process")
time_progressed_standing = 0
for episode in range(100):
    print("Running simulation for epoch", episode)
    while distance_traveled < 1000:
        if time_progressed_standing > 1000:
            break
        print(time_progressed_standing)
        driver.step()
        lidar_data = lidar.getRangeImage()
        processed_data = np.array(lidar_data).reshape(1, 200)

        action = model.predict(processed_data)[0]
        speed, steering = action

        if speed < 0:
            time_progressed_standing += basicTimeStep

        if np.isnan(speed) or np.isnan(steering):
            break

        set_speed_m_s(speed)
        set_direction_degree(steering)

        print(action)
        reward = calculate_reward(speed, lidar_data)
        print(reward)

        model.fit(processed_data, action.reshape(1, 2), epochs=episode + 1, verbose=0, sample_weight=np.array([reward]))

    emitter.send("reset".encode("utf-8"))
    distance_traveled = 0
    time_progressed_standing = 0

model.save("car_model.h5")