from rplidar import RPLidar
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from rpi_hardware_pwm import HardwarePWM
import threading
from luma.core.interface.serial import i2c
from luma.core.render import canvas
from luma.oled.device import sh1106
from gpiozero import LED, Button
from rpi_hardware_pwm import HardwarePWM
from keras.models import load_model

bp1 = Button("GPIO5")
bp2 = Button("GPIO6")
led1 = LED("GPIO17")
led2 = LED("GPIO27")
laidar_array_deg = 90

serial = i2c(port=1, address=0x3C)
device = sh1106(serial)

model_path = "pre_trainer_model.h5"
model = load_model(model_path)

with canvas(device) as draw:
    draw.rectangle(device.bounding_box, outline="white", fill="black")
    draw.text((10, 10), "Hackathon 2023", fill="white")
    draw.text((10, 20), "Cachan - CoVAPSy", fill="white")
    draw.text((10, 35), "bp2 -> demo", fill="white")
    draw.text((10, 45), "bp2 long pour arret", fill="white")

###################################################
# Initialization of motors
##################################################

stop_prop = 7.5
point_mort_prop = 0.5
vitesse_max_m_s = 3

# angle_pwm_min = 7.31 # left
angle_pwm_min = 6.5
# angle_pwm_max = 9.41 # right
angle_pwm_max = 9.81
angle_degre_max = +18  # towards left
angle_pwm_centre = 8.36  # 8.51clavier #7.65normal
angle_degre = 0

pwm_prop = HardwarePWM(pwm_channel=0, hz=50)
pwm_dir = HardwarePWM(pwm_channel=1, hz=50)
pwm_prop.start(stop_prop)
time.sleep(1.0)
pwm_prop.start(2.0)
time.sleep(0.1)
pwm_prop.start(1.0)
time.sleep(0.1)
pwm_prop.start(stop_prop)
pwm_dir.start(angle_pwm_centre)
print("PWM activated")

acqui_tableau_lidar_mm = [0] * 360  # create an array of 360 zeros
tableau_lidar_mm = [0] * 360
drapeau_nouveau_scan = False
Run_Lidar = False

a_prop = vitesse_max_m_s / (65198)
a_dir = (angle_degre_max) / (-32767)


def lidar_scan():
    global drapeau_nouveau_scan
    global acqui_tableau_lidar_mm
    global Run_Lidar
    global lidar
    print("lidar_scan task started")
    scan_avant_en_cours = False
    angle_old = 0
    while Run_Lidar == True:
        try:
            for _, _, angle_lidar, distance in lidar.iter_measures(
                    scan_type='express'):  # The array is continuously filling, so the loop is infinite
                angle = min(359, max(0, 359 - int(angle_lidar)))
                if (angle >= 260) or (angle <= 100):
                    acqui_tableau_lidar_mm[angle] = distance  # [1]: angle and [2]: distance
                if (angle < 260) and (angle > 150) and scan_avant_en_cours == True:
                    drapeau_nouveau_scan = True
                    scan_avant_en_cours = False
                if (angle >= 260) or (angle <= 100):
                    scan_avant_en_cours = True
                if Run_Lidar == False:
                    break
        except:
            print("lidar acquisition issue")


def set_direction_degre(angle_degre):
    angle_pwm = angle_pwm_centre - (angle_pwm_max - angle_pwm_min) * angle_degre / (2 * angle_degre_max)
    if angle_pwm > angle_pwm_max:
        angle_pwm = angle_pwm_max
    if angle_pwm < angle_pwm_min:
        angle_pwm = angle_pwm_min
    pwm_dir.change_duty_cycle(angle_pwm)


def set_vitesse_m_s(vitesse_m_s):
    if vitesse_m_s > vitesse_max_m_s:
        vitesse_m_s = vitesse_max_m_s
    elif vitesse_m_s < -vitesse_max_m_s:
        vitesse_m_s = -vitesse_max_m_s
    if vitesse_m_s == 0:
        pwm_prop.change_duty_cycle(stop_prop)
    elif vitesse_m_s > 0:
        vitesse = vitesse_m_s * 1.5 / 8
        pwm_prop.change_duty_cycle(stop_prop + point_mort_prop + vitesse)
    elif vitesse_m_s < 0:
        vitesse = vitesse_m_s * 1.5 / 8
        pwm_prop.change_duty_cycle(stop_prop - point_mort_prop + vitesse)


def recule():
    set_vitesse_m_s(-vitesse_max_m_s)
    time.sleep(0.2)
    set_vitesse_m_s(0)
    time.sleep(0.1)
    set_vitesse_m_s(-1)


def attente(delai_s):
    time.sleep(delai_s)


def conduite_autonome():
    global drapeau_nouveau_scan
    global acqui_tableau_lidar_mm
    global tableau_lidar_mm
    global Run_Lidar
    print("autonomous driving task started")
    while Run_Lidar == True:
        if drapeau_nouveau_scan == False:
            time.sleep(0.01)
        else:
            for i in range(-100, 101):
                tableau_lidar_mm[i] = acqui_tableau_lidar_mm[i]
            acqui_tableau_lidar_mm = [0] * 360
            print(tableau_lidar_mm)
            for i in range(-98, 99):
                if tableau_lidar_mm[i] == 0:
                    if (tableau_lidar_mm[i - 1] != 0) and (tableau_lidar_mm[i + 1] != 0):
                        tableau_lidar_mm[i] = (tableau_lidar_mm[i - 1] + tableau_lidar_mm[i + 1]) / 2
            drapeau_nouveau_scan = False

            ####################################################
            # Lucien Eldar's program with 20-degree sectors
            ###################################################

            # if tableau_lidar_mm[0] < 150 and tableau_lidar_mm[0] != 0:
            #     print("wall in front")
            #     set_direction_degre(0)
            #     recule()
            #     attente(0.5)

            # elif tableau_lidar_mm[-30] < 150 and tableau_lidar_mm[-30] != 0:
            #     print("wall on the right")
            #     set_direction_degre(-18)
            #     recule()
            #     attente(0.5)

            # elif tableau_lidar_mm[30] < 150 and tableau_lidar_mm[30] != 0:
            #     print("wall on the left")
            #     set_direction_degre(+18)
            #     recule()
            #     attente(0.5)

            # else:
            #     angle_degre = 0.02 * (tableau_lidar_mm[60] - tableau_lidar_mm[-60])  # the direction angle is the difference between the lidar radius measurements at -60 and +60°
            #     set_direction_degre(angle_degre)
            #     vitesse_m_s = 0.0006 * tableau_lidar_mm[0] + 0.1
            #     set_vitesse_m_s(vitesse_m_s)

            ###################################################

            # multiplier = 2
            # sum_right = 0
            # sum_left = 0
            # magic_value = 6000
            # counter = 0
            # mid = laidar_array_deg / 3

            # for i in range(laidar_array_deg):
                # weighting = abs(i - mid)
                # mapped_array = np.interp(weighting, (0, 60), (1, 5))

                # multiplier -= 1 / laidar_array_deg
                # if i != 0:
                    # sum_right += tableau_lidar_mm[i] * multiplier + mapped_array
                    # counter += 1

            # sum_right = sum_right / counter

            # multiplier = 2
            # counter = 0

            # for i in range(laidar_array_deg):
                # weighting = abs(i - mid)
                # mapped_array = np.interp(weighting, (0, 60), (1, 5))

                # multiplier -= 1 / laidar_array_deg
                # if i != 0:
                    # sum_left += tableau_lidar_mm[-i] * multiplier + mapped_array
                    # counter += 1

            # sum_left = sum_left / counter

            # result = sum_left - sum_right
            # if (sum_left != 0) and (sum_right != 0):
                # ratio = min(sum_left, sum_right) / max(sum_left, sum_right)
            # else:
                # ratio = 1
            # print("ratio: ", ratio)
            # ratio = (ratio ** 2) * 2
            # set_vitesse_m_s(2)
            # if sum_left > sum_right:
                # mapped_result = abs(ratio * angle_degre_max - angle_degre_max)
            # else:
                # mapped_result = -abs(ratio * angle_degre_max - angle_degre_max)
            # if mapped_result > angle_degre_max / 2:
                # mapped_result = angle_degre_max
                # vitesse_m_s = np.interp(mapped_result, (0, 18), (2, 0.1))
            # elif mapped_result < -angle_degre_max / 2:
                # mapped_result = -angle_degre_max
                # vitesse_m_s = np.interp(mapped_result, (-18, 0), (0.1, 2))
            # else:
                # vitesse_m_s = 0.0006 * tableau_lidar_mm[0] + 0.1

            action = model.predict(processed_data)[0]
            speed, steering = action

            speed_m_s = speed / 3.6
            steering_degrees = steering * (180 / 3.141)

            set_vitesse_m_s(speed_m_s)
            set_direction_degre(steering_degrees)

            # set_vitesse_m_s(vitesse_m_s)
            # set_direction_degre(-mapped_result)

            ###################################################


while True:
    while (bp1.is_pressed) or (bp2.is_pressed):
        pass

    while (not bp1.is_pressed) and (not bp2.is_pressed):
        led1.on()
        time.sleep(0.25)
        led1.off()
        time.sleep(0.25)

    if bp2.is_pressed:

        # lidar connection and start
        lidar = RPLidar("/dev/ttyUSB0", baudrate=256000)
        lidar.connect()
        print(lidar.get_info())
        lidar.start_motor()
        time.sleep(2)

        Run_Lidar = True
        thread_scan_lidar = threading.Thread(target=lidar_scan)
        thread_scan_lidar.start()
        time.sleep(1)

        thread_conduite_autonome = threading.Thread(target=conduite_autonome)
        thread_conduite_autonome.start()

        while True:
            try:
                time.sleep(1)
                if bp2.is_pressed:
                    Run_Lidar = False
                    break
            except KeyboardInterrupt:
                print("program stopped")
                Run_Lidar = False
                break

        thread_conduite_autonome.join()
        thread_scan_lidar.join()

        # lidar stop and disconnect
        lidar.stop_motor()
        lidar.stop()
        time.sleep(1)
        lidar.disconnect()
        pwm_prop.stop()
        pwm_dir.stop()
        print("PWM stopped")

        # displaying acquired data on the environment
        # print(len(tableau_lidar_mm))
        # print(tableau_lidar_mm)
        # teta = [0]*360
        # for i in range(360):
        #     teta[i] = i * np.pi / 180
        # fig = plt.figure()
        # ax = plt.subplot(111, projection='polar')
        # line = ax.scatter(teta, tableau_lidar_mm, s=5)
        # line.set_array(tableau_lidar_mm)
        # ax.set_rmax(3000)
        # ax.grid(True)
        # plt.show()