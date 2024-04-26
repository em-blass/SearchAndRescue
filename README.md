https://www.dropbox.com/scl/fi/6bqu2hz0390ma8ae4buw7/tennisdataset.zip?rlkey=t95lzsri29u5u3k70xackgbmb&st=1ibz6fbn&dl=0

## DJI Tello SAR

This project demonstrates how to use a DJI Tello drone to locate objects using TensorFlow Lite. The project uses Python for drone control and machine learning to process images captured by the drone's camera, identifying the presence of a specified object within the camera's field of view.

## Project Overview

The goal of this project is to enable a DJI Tello drone to autonomously locate pre-selected items in a defined area. It uses Python to interface with the drone and TensorFlow Lite to handle real-time image recognition.

## Prerequisites

- Python 3.7 or higher
- DJI Tello Drone

## Program Structure Overview
locate_items.py
- This is the main script that combines drone control and image recognition functionalities.

drone_control.py
- Handles all the interactions with the DJI Tello drone, including movement controls and camera handling.

image_recognition.py
- Manages the loading, processing, and recognition of images using TensorFlow Lite.

utils.py
- Contains helper functions for image processing and other utility tasks.
