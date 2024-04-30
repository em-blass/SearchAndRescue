'''
First We Need to get our necessary imports **
'''
from djitellopy import Tello
import time, cv2
from keras import models
import numpy as np

'''
Now lets connect to our drone using connect and stream on
'''
tello = Tello()
tello.connect()
tello.streamon()
time.sleep(5)

# We want to set our fly_flag to false when debugging camera/algs
fly_flag = True
# We set a mex flight time to use later in our code
max_fly_time = 500

# Set how big the camera box should be
maxW = 640
maxH = 480
# Set the font we want to appear on our Camera Box!!
font = cv2.FONT_HERSHEY_SIMPLEX
# Now we can load in our tenis ball detector model so later we can use it for predictions
model = models.load_model('tennis_ball_detector_model.h5')

# Define min window size to be recognized as a face
minW = 0.1*(maxW)
minH = 0.1*(maxH)

# Set the threshold to properly identify our tennis ball
threshold = 0.5

if fly_flag:
    tello.send_read_command("command")
    tello.send_read_command("takeoff")
    time.sleep(2)
    tello.move_up(80)

'''
This function is where we will be preforming the logic to make a prediction on finding the ball through the
camera of our drone
'''
def locate_ball():
    # start timer
    current_time = time.time()
    while True:
    # this is where we will be getting our frame data in from our camera
        img = tello.get_frame_read().frame
        small_frame = cv2.resize(img, (128, 128))
        small_frame = np.expand_dims(small_frame, axis=0)
        small_frame = small_frame / 255.0

        # Here is where we will have our model make our prediction to see if we have a tenis ball or not
        prediction = model.predict(small_frame)[0][0]
        print(f"Prediction score: {prediction}")

        # So if our prediction is greater than our threshold that means the tenis ball has been located
        if prediction > threshold:
            text = 'Tennis Ball Detected'
            # If we find the tennis ball update our time
            current_time = time.time()
        else:
            text = 'Tennis Ball Not Found'

        # This is just to display the text on screen
        cv2.putText(img, text, (10, 30), font, 1, (0, 255, 0), 2)

        # Then displaying that actual video
        cv2.imshow("Tello Stream", img)

        # Break the loop if 'q' is pressed or if no detection for 5 seconds
        if cv2.waitKey(30) == ord('q') or time.time() - current_time > 5:
            break

    cv2.destroyAllWindows()
    return prediction > threshold
'''
This scout() function makes the drone move up and down in looking for a drone
in this function we call the locate_ball() function
'''
def scout():
    # we get the current time
    start_time = time.time()
    # we check to make sure the drone isn't "idle" flying for too long
    while time.time() - start_time < max_fly_time:
        # Look for ball in the locate_ball() function
        found_ball = locate_ball()

        # check to see if a ball was found
        if found_ball:
            print("Tennis ball found!")
            # we land the drone as it is not looking for the ball anymore
            tello.land()
            break
        else:
            # if we don't find the ball we move the drone down and check again
            print("Tennis ball not found. Moving down...")
            tello.move_down(20)
            # call sleep so the drone has enough time to stabilize after moving down
            time.sleep(2)

        # Check if the drone is too close to the ground
        if tello.get_distance_tof() < 30:
            # land the drone as it is too close to the ground
            print("Drone is too close to the ground. Landing...")
            tello.land()
            break

# start searching for the tennis ball with the drone
scout()
