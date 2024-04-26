# DJI Tello SAR: Tennis Ball Detection with TensorFlow Lite

This project utilizes a DJI Tello drone equipped with TensorFlow Lite to autonomously locate and identify tennis balls using real-time image processing. The application, designed in Python, combines drone flight control with advanced machine learning techniques to recognize objects within the camera's field of view.

## Project Overview

The main objective of this project is to enable the DJI Tello drone to autonomously detect tennis balls in a predefined area using a TensorFlow Lite model. The project incorporates Python scripts for controlling the drone's movement and handling image recognition tasks.

## Prerequisites

- Python 3.7 or higher
- DJI Tello Drone
- TensorFlow, Keras, OpenCV, and djitellopy library

## Project Structure

- **drone_control.py**: Manages the drone's flight controls and streams the video for object detection.
- **detect_tennis_ball.py**: Trains a machine learning model using the MobileNetV2 architecture and performs real-time tennis ball detection.

### Key Features

1. **Autonomous Navigation**: The drone can autonomously scout an area and adjust its position to better locate tennis balls.
2. **Real-Time Image Recognition**: Utilizes a pre-trained MobileNetV2 model fine-tuned on tennis ball images to detect the presence of tennis balls in real time.

## How It Works

- **Training the Model**: The `detect_tennis_ball.py` script trains a TensorFlow model using images of tennis balls. The training and validation data are sourced from the `tennisdataset`, which should be placed in the project directory.
- **Flight and Detection**: The `drone_control.py` script controls the drone's flight and uses the trained model to detect tennis balls during flight. When a tennis ball is detected, the drone adjusts its flight path to follow the detected object.

## Usage

1. Ensure the Tello drone is charged and in an open area.
2. Run the `detect_tennis_ball.py` script to train the model. Ensure the dataset is correctly placed in the project directory.
3. Execute the `drone_control.py` to start the drone's flight and detection sequence.

## Conclusion

This project demonstrates the powerful combination of drone technology and machine learning for object detection tasks. By leveraging TensorFlow Lite and the DJI Tello drone, we can effectively identify specific objects like tennis balls in various environments.
