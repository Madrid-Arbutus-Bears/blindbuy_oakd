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
    def __init__(self,stepper_name,pins,sensor_pin,negate_sensor,reverse,max):
        self.stepper_name=stepper_name
        self.pins=pins
        self.sensor_pin=sensor_pin
        self.negate_sensor=negate_sensor
        self.reverse=reverse
        self.actual_position=0
        self.backsteps=1000 #If sensor is pressed move this steps back
        self.seq=[
            [1,0,0,0],
            [1,1,0,0],
            [0,1,0,0],
            [0,1,1,0],
            [0,0,1,0],
            [0,0,1,1],
            [0,0,0,1],
            [1,0,0,1]
        ]
        self.direction=-1 #1:CounterClockwise -1:Clockwise
        self.min=0 #Min is 0 when hit the sensor
        self.max=max #Max value to turn
            
        #Use board layout to set pin name
        GPIO.setmode(GPIO.BOARD)

        #Configure GPIO
        for pin in self.pins:
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, 0)
        
        GPIO.setup(sensor_pin,GPIO.IN)
        #GPIO.setup(sensor_pin,GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
        #Init position stepper
        self.init_position()
    
    #starts the initialisation process:
	#case 1: sensor in normal position: simply move in trigger direction till reached
	#case 2: sensor is clicked: go back to free sensor, then case 1
    def init_position(self):
        print('sensorPin: '+str(self.sensor_pin)+' '+str(GPIO.input(self.sensor_pin)))
        print('reverseInit: '+str(self.reverse))
        print('negateSensor: '+str(self.negate_sensor))
        
        #Case 2 (sensor clicked):
        if GPIO.input(self.sensor_pin):
            for i in range(1,self.backsteps):
                self.actual_position+=self.direction
                self.step()

        #Case 1 (move to sensor):
        while not GPIO.input(self.sensor_pin):
            self.actual_position-=self.direction
            self.step()

        print('Set home position')
        self.actual_position=0

    #Set goalposition
    def set_position(self,position):
        if position<self.min:
            position=self.min
            print('Set position to min: '+str(position))
        elif position>self.max:
            position=self.max
            print('Set position to max: '+str(position))
        return int(position)

    #Move one step 
    def step(self, wait=0.001):
        for pin in range(4):
                GPIO.output(self.pins[pin], self.seq[self.actual_position%8][pin])
        time.sleep(wait)

    #updates actualPosition one step towards goalposition and calls the step() method to move
    def move(self, position):
        if self.actual_position<self.set_position(position):
            self.actual_position+=1
            self.step()
        elif self.actual_position>self.set_position(position):
            self.actual_position-=1
            self.step()
        
    #Shutdown GPIO output to save energy
    def off(self):
        for pin in range(4):
            GPIO.output(self.pins[pin], 0)

stepper=Stepper(stepper_name="pan_motor", pins=[12,16,18,22], sensor_pin=36, negate_sensor=False, reverse=False, max=900)


stepper.move(900)