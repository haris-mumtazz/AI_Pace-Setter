> Python version: 3.7 \
This README was last modified on __July 31st, 2023__
# AI_Pace-Setter

## Overview

An AI pace setter or object identifier using Jetson Nano. Firstly, the camera will capture photos from the webcam. Object detection of images will be done using the TensorFlow framework. If an obstacle is on the right, the screen will prompt you to turn left and vice versa.

## Table of Contents

1. [Devices Needed](#devices-needed)
2. [Step By Step Guide](#step-by-step-guide)

## Devices Needed

- Jetson Nano
- Camera
- Monitor
- Keyboard & Mouse
- Mini HDMI converter
- USB OTG cable (Micro USB to USB)
- Micro USB power source

## Step By Step Guide

### Install the necessary Libraries and Repository
```console
sudo apt-get update
sudo apt-get install git cmake libpython3-dev python3-numpy python-pip
git clone https://github.com/haris-mumtazz/AI_Pace-Setter
```
### Run the project
#### 1. Open terminal
   
#### 2. Check Camera ID
```console
ls /dev/video*
```
It will return like '/dev/video0' which is the id of the camera connected
or '/dev/video0 /dev/video1' if 2 cameras connected

Then copy and paste the id e.g. '/dev/video0' to line 19 in main.py if you use USB CAM;
or if you use CSI CAM, just comment line 19, and uncomment line 21 (default).

#### 3. RUN
```console
cd Desktop/AI-pace-setter-main
source IOTenv/bin/activate
python main.py
```
#### 4. TROUBLESHOOTING

##### a. If you get 'nonetype' error message:

```console
deactivate
source IOTenv/bin/activate
python main.py
```

##### b. If you still get error, reboot the jetson nano and restart
```console
deactivate
sudo reboot
```
