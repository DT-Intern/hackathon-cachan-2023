import numpy as np
import tensorflow as tf
from vehicle import Driver
from controller import Lidar, Robot
from network import QNetwork
from agent import QAgent

# supervisor_node = supervisor.getFromDef("test")
# print(supervisor_node)

# Constants for the Q-Agent
input_size = 200
output_size = 2  # for target speed and rotation angle
hidden_size = 64
learning_rate = 0.001
gamma = 0.99
buffer_size = 1000
batch_size = 64
update_frequency = 10

# Create the Q-Network and target Q-Network
q_network = QNetwork(input_size, hidden_size, output_size)
target_network = QNetwork(input_size, hidden_size, output_size)
target_network.set_weights(q_network.get_weights())

driver = Driver()
emitter = driver.getDevice("emitter")

basicTimeStep = int(driver.getBasicTimeStep())
sensorTimeStep = 4 * basicTimeStep

lidar = Lidar("lidar")
lidar.enable(sensorTimeStep)
lidar.enablePointCloud()

# Initialize the Q-Agent with its training parameters
agent = QAgent(q_network, target_network, buffer_size, batch_size, learning_rate, gamma)

total_episodes = 1000
max_steps = 200
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01

traveled_distance = 0
route_progress = 0
max_speed = 28
max_turning_degree = 16

def collision_occurred(lidar_data, threshold_distance):
    for distance in lidar_data:
        if distance < threshold_distance:
            return True
    return False


def calculate_reward(speed, lidar_data):
    speed_weight = 0.7
    progress_reward = 0.3
    collision_penalty = -10.0

    # Check for collision
    if collision_occurred(lidar_data, 0.1):
        return collision_penalty

    # Calculate the overall reward
    return (speed_weight * speed) + progress_reward


def set_speed_m_s(speed_m_s):
    speed = speed_m_s * 3.6
    if speed > max_speed:
        speed = max_speed
    if speed < 0:
        speed = 0
    driver.setCruisingSpeed(speed)
    return speed


def set_direction_degree(angle_degree):
    if angle_degree > max_turning_degree:
        angle_degree = max_turning_degree
    elif angle_degree < -max_turning_degree:
        angle_degree = -max_turning_degree
    angle = angle_degree * 3.14 / 180
    driver.setSteeringAngle(angle)
    return angle

lidar_data = lidar.getRangeImage()

while driver.step() == -1:
    pass

# Get the lidar data
lidar_data = lidar.getRangeImage()

# Check if lidar data is available
if lidar_data is not None:
    # Reset the episode-specific variables
    episode_reward = 0
    episode_loss = 0

    # Get the initial state from the lidar
    state = np.array(lidar_data)

    # Add the episode variable here
    for episode in range(total_episodes):
        for step in range(max_steps):
            # Select action using epsilon-greedy policy
            driver.step()
            speed, angle = agent.get_action(state, epsilon)

            # Set the speed and angle
            speed = set_speed_m_s(speed)
            angle = set_direction_degree(angle)

            # Update the lidar data for the next step
            lidar_data = lidar.getRangeImage()
            next_state = np.array(lidar_data)

            # Calculate reward based on current state and action
            reward = calculate_reward(speed, lidar_data)

            traveled_distance += speed * 0.001
            done = traveled_distance > 1000

            # Store experience in replay buffer
            agent.replay_buffer.append((state, speed, angle, reward, next_state, done))

            if step % update_frequency == 0:
                agent.update_target_network()

            # Train the Q-Network
            loss = agent.train()

            # Update the episode-specific variables
            episode_reward += reward
            if loss is not None:
                episode_loss += loss

            state = next_state

        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        print("Resetting simulation")
        emitter.send("reset".encode("utf-8"))
        print("Episode:", episode, "Total Reward:", episode_reward, "Loss:", episode_loss)

        q_network.save_weights("q_network.h5")