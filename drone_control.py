from djitellopy import Tello
import time

tello = Tello()
tello.connect()
tello.streamon()
time.sleep(5)

tello.send_read_command("command")
tello.send_control_command("takeoff")

time.sleep(2)
def make_square():
    for i in range(2):
        tello.rotate_counter_clockwise(90)
        tello.move_forward(50)
        # check for ball()
        tello.rotate_counter_clockwise(90)
        tello.move_forward(50)
        # check for ball ()



### if image detect == true VV do this
'''
        if nameid == 'AndyNere':
            if True: #(counter > 20):

                #Check how car the X is off of the center
                lrdelta = maxW//2 - (x + w//2)

                # Rotate either L/R if we're not close to center
                if(lrdelta > .2*maxW):
                    if fly_flag:
                        rV = -30

                elif(lrdelta < -.2*maxW):
                    if fly_flag:
                        rV = 30
                #Check how far Y is off center, move up or down
                uddelta = maxH//2 - (y + h//2)

                if(uddelta > 0.2*maxH):
                    if fly_flag:
                        vV = 20

                elif(uddelta < -.2*maxH):
                    if fly_flag:
                        vV = -20

                #Check if face box is too big or too small, move back/forth
                if(w < 100):
                    if fly_flag:
                        #tello.move_forward(20)
                        dV = 15
                elif(w>200):
                    if fly_flag:
                        #tello.move_back(20)
                        dV = -15

            #draw bounding boxes
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        else:
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)

        cv2.putText(img, str(nameid), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)

'''
tello.move_up(60)
make_square()
print("height:", tello.get_height())
tello.move_down(20)
make_square()
print("height:", tello.get_height())
tello.move_down(20)
make_square()

tello.land()
