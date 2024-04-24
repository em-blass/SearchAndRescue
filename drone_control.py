## https://tello.oneoffcoder.com/python-auto-flight.html
from djitellopy import Tello

tello = Tello()
tello.connect()
tello.takeoff()

def make_square():
    for i in range(2):
        tello.move_forward(50)  # Moves forward 30cm
        tello.rotate_counter_clockwise(90)  # rotate 90 degrees counterclockwise
        tello.move_forward(50)  # Moves forward 50cm
        tello.rotate_counter_clockwise(90)  # rotate 90 degrees counterclockwise


make_square()
tello.move_down(10)
make_square()
tello.move_down(10)
make_square()
tello.land()
