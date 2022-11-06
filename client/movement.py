from enum import Enum

class Movement(Enum):
    nothing = 0
    forward = 1
    right = 2
    left = 3
    backward = 4
    

def handle_movement(x : int):
    x = Movement(x)
    if x == Movement.nothing:
        ...
    elif x == Movement.forward:
        print("forward")
    elif x == Movement.backward:
        print("backward")
    elif x == Movement.right:
        print("right")
    elif x == Movement.left:
        print("left")