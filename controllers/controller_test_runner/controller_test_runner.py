from vehicle import Driver
from controller import Lidar
from keras.models import Sequential, load_model
from keras.optimizers.legacy import Adam
from keras.layers import Dense, Dropout, BatchNormalization
import numpy as np
import random
import math

model_path = "../controller_pre_trainer/pre_trainer_model.h5"
pre_trainer_model = load_model(model_path)

model = Sequential()
model.add(Dense(64, input_shape=(200,), activation="relu"))
model.add(BatchNormalization())
model.add(Dense(32, activation="relu"))
model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(Dense(2))
model.compile(loss="mean_squared_error", optimizer=Adam(learning_rate=0.0001))
model.set_weights(pre_trainer_model.get_weights())

driver = Driver()
basicTimeStep = int(driver.getBasicTimeStep())
sensorTimeStep = 4 * basicTimeStep

emitter = driver.getDevice("emitter")
receiver = driver.getDevice("receiver")
receiver.enable(64)
emitter.send("checkpoints".encode("utf-8"))

lidar = Lidar("lidar")
lidar.enable(sensorTimeStep)
lidar.enablePointCloud()

checkpoints = []
current_checkpoint = 1
reward = 0
replay_buffer = []
batch_size = 16

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
    total_distance = euclidian_distance([old_x, old_y], [x, y])

    # print(checkpoint_distance)
    if checkpoint_distance < 0.2:
        print("Checkpoint reached")
        current_checkpoint = current_checkpoint + 1

    return (current_checkpoint - 1) + (total_distance - checkpoint_distance) / total_distance


while driver.step() != -1:
    for i in range(200):
        while True:
            driver.step()
            checkpoint_reward = 0
            if receiver.getQueueLength() > 0:
                message = receiver.getData().decode("utf-8")
                if "c" in message:
                    checkpoints.append(message.replace("c", ""))
                elif "p" in message:
                    checkpoint_reward = update_checkpoints(message.replace("p", ""))
                    print(checkpoint_reward)
                receiver.nextPacket()

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

            driver.setCruisingSpeed(float(speed))
            driver.setSteeringAngle(float(steering))

            reward = checkpoint_reward #(speed, lidar_data, checkpoint_reward)
            print(reward)

            if min(lidar_data) < 0.15:
                reward = -1000

            # print(distance_traveled)
            replay_buffer.append((processed_data, action, reward))

            if len(replay_buffer) >= batch_size:
                batch_indices = np.random.choice(len(replay_buffer), size=batch_size, replace=False)
                batch_data = [replay_buffer[i] for i in batch_indices]
                x_batch = np.concatenate([data[0] for data in batch_data])
                y_batch = np.array([data[1] for data in batch_data])

                model.fit(x_batch, y_batch, epochs=1, verbose=0)

            if min(lidar_data) < 0.15:
                emitter.send("reset".encode("utf-8"))
                current_checkpoint = 1
                break