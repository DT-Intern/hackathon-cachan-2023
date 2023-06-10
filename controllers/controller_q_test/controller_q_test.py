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

# Reinforcement learning parameters
exploration_rate = 1.0
exploration_decay = 0.995
discount_factor = 0.95

# Reward components weights
checkpoint_progress_weight = 100
collision_penalty_weight = -1000
speed_reward_weight = -0.1
smooth_steering_reward_weight = -0.1
forward_progress_reward_weight = 1
target_speed = 5

def euclidean_distance(vector_a, vector_b):
    return math.sqrt(math.pow(vector_b[0] - vector_a[0], 2) + math.pow(vector_b[1] - vector_a[1], 2))


def update_checkpoints(position):
    global current_checkpoint
    if len(checkpoints) < 2:
        return 0

    pos_x, pos_y, pos_z = map(float, position.split(", "))

    id, old_x, old_y, old_z = map(float, checkpoints[current_checkpoint - 1].split(","))
    id, x, y, z = map(float, checkpoints[current_checkpoint].split(", "))
    checkpoint_distance = euclidean_distance([x, y], [pos_x, pos_y])
    total_distance = euclidean_distance([old_x, old_y], [x, y])

    if checkpoint_distance < 0.2:
        print("Checkpoint reached")
        current_checkpoint += 1

    return (current_checkpoint - 1) + (total_distance - checkpoint_distance) / total_distance


# Training loop
while driver.step() != -1:
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

    # Use epsilon-greedy for action selection
    # if random.random() < exploration_rate:
        # action = np.random.uniform(-1, 1, size=2)
    # else:
        # action = model.predict(processed_data)[0]

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
    driver.step()

    collision_penalty = 0
    if min(lidar_data) < 0.15:
        collision_penalty = collision_penalty_weight

    # Calculate reward components
    checkpoint_progress_reward = checkpoint_reward * checkpoint_progress_weight
    collision_penalty = collision_penalty
    speed_reward = speed_reward_weight * (target_speed - abs(speed - target_speed))
    smooth_steering_reward = smooth_steering_reward_weight * abs(steering)
    forward_progress_reward = forward_progress_reward_weight * speed  # Reward for forward progress

    # Calculate the total reward
    reward = (
            checkpoint_progress_reward +
            collision_penalty +
            speed_reward +
            smooth_steering_reward +
            forward_progress_reward
    )

    #reward = 0
    #if min(lidar_data) < 0.15:
        #reward = -1000

    # checkpoint_reward = update_checkpoints(driver.getPosition())
    #reward += checkpoint_reward

    replay_buffer.append((processed_data, action, reward))

    if len(replay_buffer) >= batch_size:
        # Randomly sample from replay buffer
        batch_indices = np.random.choice(len(replay_buffer), size=batch_size, replace=False)
        batch_data = [replay_buffer[i] for i in batch_indices]
        x_batch = np.concatenate([data[0] for data in batch_data])
        y_batch = np.array([data[1] for data in batch_data])

        # Update the Q-values using Q-learning equation
        q_values = model.predict(x_batch)
        next_q_values = model.predict(x_batch)
        max_next_q_values = np.max(next_q_values, axis=1)
        max_next_q_values = max_next_q_values.reshape((-1, 1))  # Reshape to match y_batch shape
        target_q_values = y_batch + discount_factor * max_next_q_values
        q_values[np.arange(batch_size), :] = target_q_values

        # Train the model
        model.fit(x_batch, q_values, epochs=1, verbose=0)

    # Decay exploration rate
    exploration_rate *= exploration_decay

    # Reset when collision occurs
    if min(lidar_data) < 0.15:
        current_checkpoint = 1
        replay_buffer.clear()
        emitter.send("reset".encode("utf-8"))
