from vehicle import Driver
from controller import Lidar
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
import numpy as np
from keras.callbacks import EarlyStopping

pre_trainer_model = Sequential()
pre_trainer_model.add(Dense(64, input_shape=(200,), activation="relu"))
pre_trainer_model.add(BatchNormalization())
pre_trainer_model.add(Dense(32, activation="relu"))
pre_trainer_model.add(Dropout(0.1))
pre_trainer_model.add(BatchNormalization())
pre_trainer_model.add(Dense(2))
pre_trainer_model.compile(loss="mean_squared_error", optimizer="adam")

driver = Driver()

basicTimeStep = int(driver.getBasicTimeStep())
sensorTimeStep = 4 * basicTimeStep

# Lidar
lidar = Lidar("lidar")
lidar.enable(sensorTimeStep)
lidar.enablePointCloud()

speed = 10
maxSpeed = 28

angle = 0
maxangle = 0.28

driver.setSteeringAngle(angle)
driver.setCruisingSpeed(speed)

# Create empty lists to store training data
training_data = []
target_data = []

running = True
counter = 0

while driver.step() != -1:
    while running:
        driver.step()
        lidar_data = lidar.getRangeImage()

        speed = 2
        angle = lidar_data[160] - lidar_data[40]

        # clamp speed and angle to max values
        if speed > maxSpeed:
            speed = maxSpeed
        elif speed < -1 * maxSpeed:
            speed = -1 * maxSpeed
        if angle > maxangle:
            angle = maxangle
        elif angle < -maxangle:
            angle = -maxangle

        # Add current lidar_data, speed, and angle to training_data and target_data lists
        training_data.append(lidar_data)
        target_data.append([speed, angle])
        # print([speed, angle])

        driver.setCruisingSpeed(speed)
        driver.setSteeringAngle(angle)
        print("Collecting data", len(training_data))

        # Once you have collected enough training samples, you can start training the model
        if len(training_data) >= 1400:
            training_data = np.array(training_data)
            target_data = np.array(target_data)

            # Add early stopping callback
            # early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

            # Train the model with early stopping
            pre_trainer_model.fit(training_data, target_data, epochs=20, batch_size=32, validation_split=0.2)

            # Clear the training data lists for the next iteration
            training_data = []
            target_data = []

            # Save the model
            print("Saving model")
            pre_trainer_model.save("pre_trainer_model.h5")