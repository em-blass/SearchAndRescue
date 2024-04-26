from djitellopy import Tello
import time, cv2
from keras import models

tello = Tello()
tello.connect()
tello.streamon()
time.sleep(5)

fly_flag = True  # set to false when debugging camera/algs
max_fly_time = 500  # total number seconds fly time, before auto land
maxW = 640
maxH = 480
font = cv2.FONT_HERSHEY_SIMPLEX
model = models.load_model('tennis_ball_detector_model.h5')
tennis_cascade = cv2.CascadeClassifier('tennis_ball_detector_model.h5')

cap = cv2.VideoCapture(0)
_ , frame = cap.read()

# Define min window size to be recognized as a face
minW = 0.1 * (maxW)
minH = 0.1 * (maxH)

tello.send_read_command("command")
tello.send_control_command("takeoff")

time.sleep(2)


def locate_ball():
    hV = vV = dV = rV = 0
    tennis_ball = tennis_cascade.detectMultiScale(
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )
    for (x1, y, w, h) in tennis_ball:
        small_frame = cv2.resize(frame, (128, 128))
        small_frame = np.expand_dims(small_frame, axis=0)
        small_frame = small_frame / 255.0

        # Predict
        prediction = model.predict(small_frame)[0][0]
        print(f"Prediction score: {prediction}")

        if prediction > threshold:
            check = True
            text = 'Tennis Ball Detected'
        else:
            check = False

        while check:
            img = tello.get_frame_read().frame
            if True:  # (counter > 20):
                print('got here')
                tello.land()

                # Check how car the X is off of the center
                lrdelta = maxW // 2 - (x1 + w // 2)

                # Rotate either L/R if we're not close to center
                if (lrdelta > .2 * maxW):
                    if fly_flag:
                        rV = -30

                elif (lrdelta < -.2 * maxW):
                    if fly_flag:
                        rV = 30
                # Check how far Y is off center, move up or down
                uddelta = maxH // 2 - (y + h // 2)

                if (uddelta > 0.2 * maxH):
                    if fly_flag:
                        vV = 20

                elif (uddelta < -.2 * maxH):
                    if fly_flag:
                        vV = -20

                # Check if face box is too big or too small, move back/forth
                if (w < 100):
                    if fly_flag:
                        # tello.move_forward(20)
                        dV = 15
                elif (w > 200):
                    if fly_flag:
                        # tello.move_back(20)
                        dV = -15

            # draw bounding boxes
            cv2.rectangle(img, (x1, y), (x1 + w, y + h), (0, 255, 0), 2)

        cv2.putText(img, text, (x1 + 5, y - 5), font, 1, (255, 255, 255), 2)
        # cv2.putText(img, str(prediction), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)


def scout():
    ## look for ball
    locate_ball()
    # if we dont find it
    tello.move_down(15)
    time.sleep(5)
    scout()


tello.move_up(80)
print("height:", tello.get_height())
time.sleep(5)
scout()

tello.land()
