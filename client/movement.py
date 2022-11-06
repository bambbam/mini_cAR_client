from enum import Enum
import RPi.GPIO as GPIO
import time

SW = (5, 6, 13, 19)
DIR = (("go", "w"), ("right", "d"), ("left", "a"), ("back", "s"))
CTRL = ((0, 1, 0, 1), (0, 1, 1, 0), (1, 0, 0, 1), (1, 0, 1, 0))
PWMA = 18
AIN1 = 22
AIN2 = 27
PWMB = 23
BIN1 = 25
BIN2 = 24

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
for i in range(4):
    GPIO.setup(SW[i], GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(PWMA, GPIO.OUT)
GPIO.setup(AIN1, GPIO.OUT)
GPIO.setup(AIN2, GPIO.OUT)
GPIO.setup(PWMB, GPIO.OUT)
GPIO.setup(BIN1, GPIO.OUT)
GPIO.setup(BIN2, GPIO.OUT)

L_Motor = GPIO.PWM(PWMA, 500)
L_Motor.start(0)
R_Motor = GPIO.PWM(PWMB, 500)
R_Motor.start(0)




class Movement(Enum):
    nothing = 0
    forward = 1
    right = 2
    left = 3
    backward = 4
    
def move(x : Movement):
    if x==Movement.nothing:
        return
    i = x.value
    GPIO.output(AIN1, CTRL[i][0])
    GPIO.output(AIN2, CTRL[i][1])
    L_Motor.ChangeDutyCycle(50)
    GPIO.output(BIN1, CTRL[i][2])
    GPIO.output(BIN2, CTRL[i][3])
    R_Motor.ChangeDutyCycle(50)

    time.sleep(0.1)

def handle_movement(x : int):
    x = Movement(x)
    move(x)