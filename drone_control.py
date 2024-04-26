from djitellopy import Tello
import time, cv2
from keras import models
import numpy as np

tello = Tello()
tello.connect()
tello.streamon()
time.sleep(5)

fly_flag = False  # set to false when debugging camera/algs
max_fly_time = 500  # total number seconds fly time, before auto land

maxW = 640
maxH = 480
font = cv2.FONT_HERSHEY_SIMPLEX

model = models.load_model('tennis_ball_detector_model.h5')

# Define min window size to be recognized as a face
minW = 0.1 * maxW
minH = 0.1 * maxH

threshold = 0.5  # Adjust this value according to your model's performance

tello.send_read_command("command")
tello.send_control_command("takeoff")
time.sleep(2)

def locate_ball():
    while True:
        img = tello.get_frame_read().frame
        small_frame = cv2.resize(img, (128, 128))
        small_frame = np.expand_dims(small_frame, axis=0)
        small_frame = small_frame / 255.0
        
        # Predict
        prediction = model.predict(small_frame)[0][0]
        print(f"Prediction score: {prediction}")
        
        if prediction > threshold:
            text = 'Tennis Ball Detected'
        else:
            text = 'Tennis Ball Not Found'
        
        # Display the text on the frame
        cv2.putText(img, text, (10, 30), font, 1, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow("Tello Stream", img)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()

def scout():
    ## look for ball
    locate_ball()
    
    # if we dont find it
    tello.move_down(15)
    time.sleep(5)
    scout()
    
    print("height:", tello.get_height())
    time.sleep(5)
    scout()
    
    tello.land()

scout()
