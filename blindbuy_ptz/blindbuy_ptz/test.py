#sudo pip install Jetson.GPIO

import Jetson.GPIO as GPIO
#import RPi.GPIO as GPIO
import time

#GPIO.setmode(GPIO.BCM)
GPIO.setmode(GPIO.BOARD)
control_pins2 = [12,16,18,22]
control_pins = [22,18,16,12]
for pin in control_pins:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, 0)
halfstep_seq = [
    [1,0,0,0],
    [1,1,0,0],
    [0,1,0,0],
    [0,1,1,0],
    [0,0,1,0],
    [0,0,1,1],
    [0,0,0,1],
    [1,0,0,1]
]
while True:
    for i in range(1000):
        for halfstep in range(8):
            for pin in range(4):
                GPIO.output(control_pins[pin], halfstep_seq[halfstep][pin])
            time.sleep(0.001) 
    for i in range(1000):
        for halfstep in range(8):
            for pin in range(4):
                GPIO.output(control_pins2[pin], halfstep_seq[halfstep][pin])
            time.sleep(0.001)
GPIO.cleanup()