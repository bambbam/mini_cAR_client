from enum import Enum
import os
if os.environ.get("mode")=="prod":
    import RPi.GPIO as GPIO
import time
import threading
from client.singleton import Singleton
import asyncio

class Movement(Enum):
    nothing = 0
    forward = 1
    right = 2
    left = 3
    backward = 4
    stop = 5
    beep = 6
    speedup = 7
    speeddown = 8

class BaseSetting(Singleton):
    t = 0
    def __init__(self):
        if BaseSetting.t == 1:
            return
        BaseSetting.t+=1
        self.SW = (5, 6, 13, 19)
        self.DIR = (("go", "w"), ("right", "d"), ("left", "a"), ("back", "s"))
        self.CTRL = ((0, 1, 0, 1), (0, 1, 1, 0), (1, 0, 0, 1), (1, 0, 1, 0))
        self.PWMA = 18
        self.AIN1 = 22
        self.AIN2 = 27
        self.PWMB = 23
        self.BIN1 = 25
        self.BIN2 = 24
        self.BUZZER = 12
        self.SPEED = 50
        self.prevMove = Movement.nothing

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
        GPIO.setup(self.BUZZER, GPIO.OUT)

        self.L_Motor = GPIO.PWM(self.PWMA, 500)
        self.L_Motor.start(0)
        self.R_Motor = GPIO.PWM(self.PWMB, 500)
        self.R_Motor.start(0)
        self.p = GPIO.PWM(self.BUZZER, 391)
    def move(self,x : Movement):
        if x==Movement.nothing:
            return
        i = x.value - 1
        GPIO.output(self.AIN1, self.CTRL[i][0])
        GPIO.output(self.AIN2, self.CTRL[i][1])
        self.L_Motor.ChangeDutyCycle(self.SPEED)
        GPIO.output(self.BIN1, self.CTRL[i][2])
        GPIO.output(self.BIN2, self.CTRL[i][3])
        self.R_Motor.ChangeDutyCycle(self.SPEED)


    def stop(self):
        GPIO.output(self.AIN1, 0)
        GPIO.output(self.AIN2, 1)
        self.L_Motor.ChangeDutyCycle(0)
        GPIO.output(self.BIN1, 0)
        GPIO.output(self.BIN2, 1)
        self.R_Motor.ChangeDutyCycle(0)
    
    def beep(self):
        for i in range(2):
            self.p.start(50)
            self.p.ChangeFrequency(391)
            time.sleep(0.2)

            self.p.stop()
            time.sleep(0.1)

def handle_movement(x : int):
    x = Movement(x)
    if x == Movement.stop:
        BaseSetting().stop()
    elif x == Movement.beep:
        BaseSetting().beep()
    elif x == Movement.speedup:
        if BaseSetting().SPEED + 10 <= 100:
            BaseSetting().SPEED += 10
            BaseSetting().move(BaseSetting().prevMove)
    elif x == Movement.speeddown:
        if BaseSetting().SPEED - 10 >= 10:
            BaseSetting().SPEED -= 10
            BaseSetting().move(BaseSetting().prevMove)
    else:
        BaseSetting().move(x)
        if x != Movement.nothing:
            BaseSetting().prevMove = x
            
def handle_movement_with_delay(x: int, delay: float = 0.0):
    handle_movement(x)
    time.sleep(delay)
    handle_movement(Movement.stop.value)

class CarController(Singleton):
    sema = threading.Semaphore(0)
    control = 0
    delay = 0
    count = 0
    
    @classmethod
    def set_control(cls, control, delay = 0.0):
        cls.control = control
        cls.delay = delay
        if cls.count == 0:
            cls.sema.release()
        cls.count += 1
    
    @classmethod
    def run(cls):
        while True: 
            cls.sema.acquire()
            print(cls.control, cls.delay)
            if os.environ.get("mode")=="prod":
                handle_movement_with_delay(cls.control, cls.delay)                
            cls.control = 0;
            cls.count = 0