# Copyright 1996-2022 Cyberbotics Ltd.
#
# Controle de la voiture TT-02 simulateur CoVAPSy pour Webots 2022b
# Inspiré de vehicle_driver_altino controller
# Kévin Hoarau, Anthony Juton, Bastien Lhopitallier, Martin Taynaud
# janvier 2023

from vehicle import Driver
from controller import Lidar, Display
import numpy as np
import matplotlib.pyplot as plt

driver = Driver()

basicTimeStep = int(driver.getBasicTimeStep())
sensorTimeStep = 4 * basicTimeStep

# Reading the lidar data
lidar = Lidar("lidar")
lidar.enable(sensorTimeStep)
lidar.enablePointCloud()

keyboard = driver.getKeyboard()
keyboard.enable(sensorTimeStep)

# Measure the speed
speed = 0
maxSpeed = 28  # km/h

# Turning and maximum turning angle
angle = 0
maxangle_degre = 16

# mise a zéro de la vitesse et de la direction
driver.setSteeringAngle(angle)
driver.setCruisingSpeed(speed)

tableau_lidar_mm = [0] * 360


def set_vitesse_m_s(vitesse_m_s):
    speed = vitesse_m_s * 3.6
    if speed > maxSpeed:
        speed = maxSpeed
    if speed < 0:
        speed = 0
    driver.setCruisingSpeed(speed)


def set_direction_degre(angle_degre):
    if angle_degre > maxangle_degre:
        angle_degre = maxangle_degre
    elif angle_degre < -maxangle_degre:
        angle_degre = -maxangle_degre
    angle = angle_degre * 3.14 / 180
    driver.setSteeringAngle(angle)


# mode auto desactive
modeAuto = False
print("cliquer sur la vue 3D pour commencer")
print("a pour mode auto (pas de mode manuel sur TT02_jaune), n pour stop")

num_ranges = lidar.getHorizontalResolution()
lidar_data = np.zeros(num_ranges)

num_points = 200
angles = np.linspace(-99, 99, num_points)
distances = np.zeros(num_points)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, polar=True)
theta = np.radians(angles)
line, = ax.plot(theta, distances, marker="o", linestyle="")
ax.set_title("LiDAR Data")
ax.set_rticks([0.5, 1, 1.5])
ax.grid(True)
ax.set_theta_zero_location("N")
plt.ion()
plt.show()

while driver.step() != -1:
    while True:
        lidar_data = lidar.getRangeImage()
        lidar_data.reverse()
        line.set_ydata(lidar_data)
        fig.canvas.draw()
        plt.pause(0.1)

        # acquisition des donnees du lidar
        # recuperation de la touche clavier
        currentKey = keyboard.getKey()
        if currentKey != -1:
            print(currentKey)

        if currentKey == -1:
            break

        elif currentKey == ord('n') or currentKey == ord('N'):
            if modeAuto:
                modeAuto = False
                print("--------Modes Auto TT-02 jaune Désactivé-------")
        elif currentKey == ord('a') or currentKey == ord('A'):
            if not modeAuto:
                modeAuto = True
                print("------------Mode Auto TT-02 jaune Activé-----------------")

    # acquisition des donnees du lidar
    donnees_lidar_brutes = lidar.getRangeImage()
    for i in range(200):
        tableau_lidar_mm[(i - 100)] = 1000 * donnees_lidar_brutes[i]

    if not modeAuto:
        set_direction_degre(0)
        set_vitesse_m_s(0)

    if modeAuto:
        ########################################################
        # Programme de Lucien et Eldar
        #######################################################

        if tableau_lidar_mm[0] < 1000:
            vitesse_m_s = 0.55
        else:
            vitesse_m_s = 0.000625 * tableau_lidar_mm[0]

            # l'angle de la direction est la différence entre les mesures des rayons
        # du lidar à -60 et +60
        angle_degre = 57 * (tableau_lidar_mm[60] - tableau_lidar_mm[-60 % 360])
        if tableau_lidar_mm[-10] < 1000 and tableau_lidar_mm[10] > 1000:
            angle_degre = +18
        if tableau_lidar_mm[10] < 1000 and tableau_lidar_mm[-10 % 360] > 1000:
            angle_degre = -18
        if tableau_lidar_mm[0] < 1000 and tableau_lidar_mm[-10 % 360] > 1000:
            angle_degre = -18
        if tableau_lidar_mm[0] < 1000 and tableau_lidar_mm[10] > 1000:
            angle_degre = +18
        if tableau_lidar_mm[0] < 250:
            print("voiture bloquée")

        if tableau_lidar_mm[-90] > 2000:
            angle_degre = -18
            print("gros virage gauche")
        elif tableau_lidar_mm[90] > 2000:
            angle_degre = +18
            print("gros virage droite")

        set_direction_degre(angle_degre)
        set_vitesse_m_s(vitesse_m_s)
    #########################################################

