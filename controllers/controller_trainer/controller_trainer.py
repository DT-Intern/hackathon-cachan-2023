import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.models import Sequential, load_model
from keras.optimizers.legacy import Adam
from keras.layers import Dense, Dropout, BatchNormalization
from vehicle import Driver
from controller import Lidar
import numpy as np
import random

time_progressed_standing = 0
reward = 0
distance_traveled = 0
prev_x = 0
prev_y = 0
previous_checkpoint_reward = 0
checkpoints = []
current_checkpoint = 1
update_target_network = 1000  # Update target network every 1000 steps
replay_buffer = []
batch_size = 32

pre_trainer_model = load_model("../controller_pre_trainer/pre_trainer_model.h5")

model = Sequential()
model.add(Dense(64, input_shape=(200,), activation="relu"))
model.add(BatchNormalization())
model.add(Dense(32, activation="relu"))
model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(Dense(2))
model.compile(loss="mean_squared_error", optimizer=Adam(learning_rate=0.0001))
model.set_weights(pre_trainer_model.get_weights())

target_model = Sequential()
target_model.add(Dense(64, input_shape=(200,), activation="relu"))
target_model.add(BatchNormalization())
target_model.add(Dense(32, activation="relu"))
target_model.add(Dropout(0.1))
target_model.add(BatchNormalization())
target_model.add(Dense(2))
target_model.compile(loss="mean_squared_error", optimizer=Adam(learning_rate=0.00001))
target_model.set_weights(model.get_weights())

driver = Driver()
emitter = driver.getDevice("emitter")
receiver = driver.getDevice("receiver")
receiver.enable(64)
emitter.send("checkpoints".encode("utf-8"))

basicTimeStep = int(driver.getBasicTimeStep())
sensorTimeStep = 4 * basicTimeStep

lidar = Lidar("lidar")
lidar.enable(sensorTimeStep)
lidar.enablePointCloud()

def collision_occurred(lidar_data, threshold_distance):
    return min(lidar_data) < threshold_distance


def calculate_reward(speed, lidar_data, checkpoint_reward):
    # Define reward weights
    # speed_weight = 0.2
    checkpoint_weight = 1.5
    collision_penalty = -100.0

    # Check for collision
    if collision_occurred(lidar_data, 0.15):
        return collision_penalty

    # Calculate checkpoint reward
    reward = checkpoint_reward * checkpoint_weight

    # Calculate speed reward
    # reward += speed * speed_weight

    # Calculate the overall reward
    return reward
    # Define reward weights
    # speed_weight = 0.4
    # checkpoint_weight = 0.8
    # travel_weight = 0.5
    # collision_penalty = -1000.0
    # reward = 0

    # Check for collision
    # if collision_occurred(lidar_data, 0.2):
        # return collision_penalty

    # if lidar_data[100] < 0.5:
        # reward -= 50

    # reward += (len(passed_checkpoints) + checkpoint_reward) * checkpoint_weight

    # reward += distance_traveled * travel_weight

    # Calculate the overall reward
    # return reward + (speed_weight * speed)


def epsilon_greedy_action(q_values, epsilon):
    if random.random() < epsilon:
        return q_values[0] + random.uniform(-epsilon, epsilon), q_values[1] + random.uniform(-epsilon, epsilon)
    else:
        return q_values


def euclidian_distance(vector_a, vector_b):
    return math.sqrt(math.pow(vector_b[0] - vector_a[0], 2) + math.pow(vector_b[1] - vector_a[1], 2))


def update_checkpoints(position):
    global current_checkpoint
    # total_distance = 0
    if len(checkpoints) < 2:
        return 0

    pos_x, pos_y, pos_z = map(float, position.split(", "))

    id, old_x, old_y, old_z = map(float, checkpoints[current_checkpoint - 1].split(","))
    id, x, y, z = map(float, checkpoints[current_checkpoint].split(", "))
    checkpoint_distance = euclidian_distance([x, y], [pos_x, pos_y])
    total_distance = euclidian_distance([old_x, old_y], [pos_x, pos_y])

    if checkpoint_distance < 0.1:
        current_checkpoint = current_checkpoint + 1

    return (current_checkpoint - 1) + (total_distance - checkpoint_distance) / total_distance

    # for i in range(len(available_checkpoints)):
        # id, x, y, z = map(float, available_checkpoints[i].split(", "))
        # checkpoint_distance = euclidian_distance([x, y], [pos_x, pos_y])
        # total_distance += checkpoint_distance

        # Add the checkpoint to the closed list
        #if checkpoint_distance < 0.17:
            # if id not in passed_checkpoints:
                # passed_checkpoints.append(id)
                # available_checkpoints.remove(available_checkpoints[i])

    # return total_distance


epsilon = 0.2  # Initial exploration rate
epsilon_min = 0.01  # Minimum exploration rate
epsilon_decay = 0.8  # Exploration rate decay factor


print("Initialization complete")

while driver.step() == -1:
    pass

print("Starting learning process")

for episode in range(1000):
    print("Running simulation for epoch", episode)
    while distance_traveled < 1000:
        checkpoint_reward = 0
        if receiver.getQueueLength() > 0:
            message = receiver.getData().decode("utf-8")
            if "c" in message:
                checkpoints.append(message.replace("c", ""))
            elif "p" in message:
                checkpoint_reward = update_checkpoints(message.replace("p", ""))
                # pos_x, pos_y, pos_z = map(float, message.replace("p", "").split(", "))
                # distance_traveled += euclidian_distance([pos_x, pos_y], [prev_x, prev_y])
                # prev_x = pos_x
                # prev_y = pos_y
            receiver.nextPacket()

        driver.step()
        lidar_data = lidar.getRangeImage()
        processed_data = np.array(lidar_data).reshape(1, 200)

        action = model.predict(processed_data)[0]
        # speed, steering = epsilon_greedy_action(action, epsilon)
        speed, steering = action

        if abs(speed) < 0.5:
            time_progressed_standing += basicTimeStep

        if np.isnan(speed) or np.isnan(steering):
            continue

        driver.setCruisingSpeed(float(speed))
        driver.setSteeringAngle(float(steering))

        reward += calculate_reward(speed, lidar_data, checkpoint_reward)
        print(reward)
        # print(distance_traveled)
        replay_buffer.append((processed_data, action, reward))

        if len(replay_buffer) >= batch_size:
            batch_indices = np.random.choice(len(replay_buffer), size=batch_size, replace=False)
            batch_data = [replay_buffer[i] for i in batch_indices]
            x_batch = np.concatenate([data[0] for data in batch_data])
            y_batch = np.array([data[1] for data in batch_data])

            model.fit(x_batch, y_batch, epochs=1, verbose=0)

        # if len(replay_buffer) >= batch_size:
            # batch_indices = np.random.choice(len(replay_buffer), size=batch_size, replace=False)
            # batch_data = [replay_buffer[i] for i in batch_indices]
            # x_batch = np.concatenate([data[0] for data in batch_data])
            # y_batch = np.array([data[1] for data in batch_data])
            # reward_batch = np.array([data[2] for data in batch_data])

            # q_values = model.predict(x_batch)
            # target_q_values = target_model.predict(x_batch)

            # for i, (_, action, reward) in enumerate(batch_data):
                # if not np.isnan(reward):
                    # q_values[i] = action + reward
                # else:
                    # q_values[i] = action

            # model.fit(x_batch, y_batch, epochs=1, verbose=0)

        if collision_occurred(lidar_data, 0.25) or time_progressed_standing > 4000:
            driver.setCruisingSpeed(0)
            driver.setSteeringAngle(0)
            break

        if driver.getTime() % update_target_network == 0:
            target_model.set_weights(model.get_weights())

        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        previous_checkpoint_reward = checkpoint_reward

    time_progressed_standing = 0
    distance_traveled = 0
    prev_x = 0
    prev_y = 0
    reward = 0
    # previous_checkpoint_reward =
    current_checkpoint = 1
    #for i in range(len(passed_checkpoints)):
        #checkpoint = passed_checkpoints[i]
        #available_checkpoints.append(checkpoint)
        #passed_checkpoints.remove(checkpoint)
    emitter.send("reset".encode("utf-8"))

model.save("car_model.h5")
