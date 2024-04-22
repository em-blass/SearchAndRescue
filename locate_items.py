# Import necessary modules
from drone_control import DroneController
from image_recognition import ImageRecognizer

def main():
    # Initialize the drone controller
    drone = DroneController()

    # Connect to the drone
    drone.connect()

    # Initialize the image recognizer
    recognizer = ImageRecognizer(model_path='model.tflite')

    # Start the drone's video stream
    drone.start_video_stream()

    # Main loop to process incoming frames
    while True:
        frame = drone.get_frame()

        # Detect objects in the frame
        detections = recognizer.detect_objects(frame)

        # Analyze detections and control drone based on results
        for detection in detections:
            if detection['label'] == 'tennis ball':
                print("Tennis ball detected:", detection)
                drone.move_towards(detection['location'])

        # Implement a safe stop condition
        if drone.should_stop():
            break

    # Cleanup on exit
    drone.stop_video_stream()
    drone.disconnect()

if __name__ == '__main__':
    main()