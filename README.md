# CVBased-Login
Code to open IIT Mandi moodle using computer vision

**1) Run the following commands to download libraries:**

import cv2

import time

import os

import torch

import numpy as np


**2)** Enter names of users you want to authorize in userauth list in user.py

**3)** replace userid and password in user.py

**4)** Add photos of Users(Authorized as well as Unauthorized(if you wish to)) in the photos
folder by creating a folder inside 'photos' with name same as the person.

**5)** (optional)After installing GeckoDriverManager from line 18 in face-detection py, comment line 18
and uncomment line 19 replacing with path of GeckoDriverManager to prevent installation each time.

# **Functioning**

**While running the program and camera, press Q to exit the camera and open the webpage.
Also you can capture and save the camera feed and enter name of the person in the camera feed to 
train the model with the person's image.**




