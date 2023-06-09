import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, BatchNormalization
from vehicle import Driver
from controller import Lidar
import numpy as np
import random

pre_trainer_model = load_model("../controller_pre_trainer/pre_trainer_model.h5")

model = Sequential()
model.add(Dense(64, input_shape=(200,), activation="relu"))
model.add(BatchNormalization())
model.add(Dense(32, activation="relu"))
model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(Dense(2))
model.set_weights(pre_trainer_model.get_weights())
model.compile(loss="mean_squared_error", optimizer="adam")

target_model = Sequential()
target_model.add(Dense(64, input_shape=(200,), activation="relu"))
target_model.add(BatchNormalization())
target_model.add(Dense(32, activation="relu"))
target_model.add(Dropout(0.1))
target_model.add(BatchNormalization())
target_model.add(Dense(2))
target_model.set_weights(model.get_weights())

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
    speed_weight = 1.2
    collision_penalty = -20.0

    # Check the distance to the wall
    distance_diff = lidar_data[0] - lidar_data[-1]
    distance_weight = 0.4

    # Check for collision
    if collision_occurred(lidar_data, 0.2):
        return collision_penalty

    # Calculate the overall reward
    return (speed_weight * speed) - distance_diff * distance_weight


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

def epsilon_greedy_action(q_values, epsilon):
    if random.random() < epsilon:
        return random.uniform(-1, 1), random.uniform(-1, 1)
    else:
        return q_values

epsilon = 1.0  # Initial exploration rate
epsilon_min = 0.01  # Minimum exploration rate
epsilon_decay = 0.995  # Exploration rate decay factor

print("Initialization complete")

while driver.step() == -1:
    pass

print("Starting learning process")
time_progressed_standing = 0
update_target_network = 1000  # Update target network every 1000 steps
replay_buffer = []
batch_size = 32

for episode in range(1000):
    print("Running simulation for epoch", episode)
    while distance_traveled < 1000:
        if time_progressed_standing > 5000:
            break
        driver.step()
        lidar_data = lidar.getRangeImage()
        processed_data = np.array(lidar_data).reshape(1, 200)

        action = model.predict(processed_data)[0]
        speed, steering = epsilon_greedy_action(action, epsilon)

        if speed < 0:
            time_progressed_standing += basicTimeStep

        if np.isnan(speed) or np.isnan(steering):
            continue

        if collision_occurred(lidar_data, 0.17):
            break

        if steering > 0.314:
            steering = 0.3
        elif steering < -0.314:
            steering = -0.3

        driver.setCruisingSpeed(float(speed))
        driver.setSteeringAngle(float(steering))

        reward = calculate_reward(speed, lidar_data)
        replay_buffer.append((processed_data, action, reward))

        if len(replay_buffer) >= batch_size:
            batch_indices = np.random.choice(len(replay_buffer), size=batch_size, replace=False)
            batch_data = [replay_buffer[i] for i in batch_indices]
            x_batch = np.concatenate([data[0] for data in batch_data])
            y_batch = np.array([data[1] for data in batch_data])
            reward_batch = np.array([data[2] for data in batch_data])

            q_values = model.predict(x_batch)
            target_q_values = target_model.predict(x_batch)

            for i, (_, action, reward) in enumerate(batch_data):
                if not np.isnan(reward):
                    q_values[i] = action + reward
                else:
                    q_values[i] = action

            model.fit(x_batch, q_values, epochs=5, verbose=0)

        if driver.getTime() % update_target_network == 0:
            target_model.set_weights(model.get_weights())

        distance_traveled += speed * basicTimeStep / 1000
        print(distance_traveled)
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

    emitter.send("reset".encode("utf-8"))
    distance_traveled = 0
    time_progressed_standing = 0

model.save("car_model.h5")
