from djitellopy import Tello
import time

tello = Tello()
tello.connect()
tello.streamon()  # Start video stream and initialize IMU
time.sleep(5)     # Wait for 5 seconds for IMU calibration

tello.send_read_command("command")   # Switch to command mode
tello.send_control_command("takeoff")  # Take off and enter joystick mode

time.sleep(2)
def make_square():
    for i in range(2):
        tello.rotate_counter_clockwise(90)
        tello.move_forward(50)
        tello.rotate_counter_clockwise(90)
        tello.move_forward(50)

tello.move_up(60)
make_square()
print("height:", tello.get_height())
tello.move_down(20)
make_square()
print("height:", tello.get_height())
tello.move_down(20)
make_square()

tello.land()