import torch
import numpy as np
from vehicle import Driver
from controller import Lidar
from network import QNetwork
from agent import QAgent

# Constants for the Q-Agent
input_size = 200
output_size = 3
hidden_size = 64
learning_rate = 0.001
gamma = 0.99
buffer_size = 1000
batch_size = 64
update_frequency = 10

# Create the Q-Network and target Q-Network
q_network = QNetwork(input_size, output_size, hidden_size)
target_network = QNetwork(input_size, output_size, hidden_size)
target_network.load_state_dict(q_network.state_dict())
target_network.eval()

driver = Driver()

basicTimeStep = int(driver.getBasicTimeStep())
sensorTimeStep = 4 * basicTimeStep

lidar = Lidar("lidar")
lidar.enable(sensorTimeStep)

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

# Main training loop
for episode in range(total_episodes):
    # Reset the simulation and the lidar
    driver.simulationReset()
    lidar.enable(sensorTimeStep)

    # Reset the episode-specific variables
    episode_reward = 0
    episode_loss = 0

    # Get the initial state form the lidar
    lidar_data = lidar.getRangeImage()
    state = np.array(lidar_data)

    for step in range(max_steps):
        # Select action using epsilon-greedy policy
        speed, angle = agent.get_action(state, epsilon)
        speed = set_speed_m_s(speed)
        angle = set_direction_degree(angle)

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
        episode_loss += loss
        state = next_state

        if done:
            break

    epsilon = max(epsilon * epsilon_decay, epsilon_min)

    print("Episode:", episode, "Total Reward:", episode_reward, "Loss:", episode_loss)

    torch.save(agent.q_network.state_dict(), "q_network.pth")
