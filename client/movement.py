from enum import Enum
import RPi.GPIO as GPIO
import time

class Singleton(object):
    _instance = None

    def __new__(class_, *args, **kwargs):
        if not isinstance(class_._instance, class_):
            class_._instance = super(Singleton, class_).__new__(class_)
        return class_._instance


class Movement(Enum):
    nothing = 0
    forward = 1
    right = 2
    left = 3
    backward = 4
class BaseSetting(Singleton):
    def __init__(self):
        self.SW = (5, 6, 13, 19)
        self.DIR = (("go", "w"), ("right", "d"), ("left", "a"), ("back", "s"))
        self.CTRL = ((0, 1, 0, 1), (0, 1, 1, 0), (1, 0, 0, 1), (1, 0, 1, 0))
        self.PWMA = 18
        self.AIN1 = 22
        self.AIN2 = 27
        self.PWMB = 23
        self.BIN1 = 25
        self.BIN2 = 24


        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        for i in range(4):
            GPIO.setup(self.SW[i], GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
        GPIO.setup(self.PWMA, GPIO.OUT)
        GPIO.setup(self.AIN1, GPIO.OUT)
        GPIO.setup(self.AIN2, GPIO.OUT)
        GPIO.setup(self.PWMB, GPIO.OUT)
        GPIO.setup(self.BIN1, GPIO.OUT)
        GPIO.setup(self.BIN2, GPIO.OUT)

        L_Motor = GPIO.PWM(self.PWMA, 500)
        L_Motor.start(0)
        R_Motor = GPIO.PWM(self.PWMB, 500)
        R_Motor.start(0)

    def move(self,x : Movement):
        if x==Movement.nothing:
            return
        i = x.value-1
        GPIO.output(self.AIN1, self.CTRL[i][0])
        GPIO.output(self.AIN2, self.CTRL[i][1])
        self.L_Motor.ChangeDutyCycle(50)
        GPIO.output(self.BIN1, self.CTRL[i][2])
        GPIO.output(self.BIN2, self.CTRL[i][3])
        self.R_Motor.ChangeDutyCycle(50)

    time.sleep(0.1)

def handle_movement(x : int):
    x = Movement(x)
    BaseSetting().move(x)



    
