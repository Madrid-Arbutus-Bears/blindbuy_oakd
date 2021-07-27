import Jetson.GPIO as GPIO
import time
#PAN
#sensor_pin=36
#pins=[12,16,18,22]


class Stepper:
    #WARNING: make sure all the parameters are set right to avoid pysical damage
	#stepper_name: String - the name of the stepper motor
	#pins: int[4] - the numbers of the motor GPIO pins 
	#sensor_pin: int - the number of the sensor GPIO pin 
	#negate_sensor: boolean - inverts the sensorinput if true 
	#reverse: boolean - inverts the initialisation direction
    def __init__(self,stepper_name,pins,sensor_pin,negate_sensor,reverse):
        self.stepper_name=stepper_name
        self.pins=pins
        self.sensor_pin=sensor_pin
        self.negate_sensor=negate_sensor
        self.reverse=reverse
        
        #Use board layout to set pin name
        GPIO.setmode(GPIO.BOARD)

        #Configure GPIO
        for pin in self.pins:
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, 0)
        
        GPIO.setup(sensor_pin,GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
        #Init position stepper
        self.init_position()
    
    #starts the initialisation process:
	#case 1: sensor in normal position: simply move in trigger direction till reached
	#case 2: sensor is clicked: go back to free sensor, then case 1
    def init_position(self):
        print('sensorPin: '+str(self.sensor_pin)+' '+str(GPIO.input(self.sensor_pin)))
        print('reverseInit: '+str(self.reverse))
        print('negateSensor: '+str(self.negate_sensor))

stepper=Stepper(stepper_name="pan_motor", pins=[12,16,18,22], sensor_pin=36, negate_sensor=False, reverse=False)