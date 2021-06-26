from datetime import datetime
import io
import time
import os
import pathlib

import cv2
import numpy as np
from PIL import Image
import picamera

# Local imports
from predict import detect_cat, predict

camera = picamera.PiCamera()
camera.resolution = (1024, 720) # 682.666
camera.framerate = 8 # 32
camera.rotation = 90
camera.iso = 1600
camera.start_preview()

stream = picamera.PiCameraCircularIO(camera, seconds=20)
camera.start_recording(stream, format='h264')

# Overlay on local display
#overlay_data = np.zeros((1024, 720, 3), dtype=np.uint8)
#overlay_data[512, :, :] = 0xff
#overlay_data[:, 360, :] = 0xff
#overlay = camera.add_overlay(np.frombuffer(overlay_data), layer=3, alpha=255)

time.sleep(1)

step_sec = 1

def date_str():
    return '{:%Y-%m-%d.%H-%M-%S}'.format(datetime.now())

prior_image = None
def detect_motion(current_image):
    global prior_image
    blur = 5

    if prior_image is None:
        prior_image = current_image
        return False, None

    diff = cv2.absdiff(prior_image, current_image)

    diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    diff = cv2.GaussianBlur(diff, (blur, blur), 0)

    _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

    dilated = cv2.dilate(thresh, None, iterations=24)

    contours, hierarchy = cv2.findContours( dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    motion = False
    (lx, ly, lw, lh) = (None, None, None, None)
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        area = w * h
        if area < 25000 or w < 200 or h < 200:
            continue

        cv2.rectangle(prior_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(prior_image, "Move {}px2".format(area), (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        motion = True
        (lx, ly, lw, lh) = (x, y, w, h)

    cropped_image = None
    if motion:
        cv2.drawContours(prior_image, contours, -1, (0,255,0), 2)

        dstr = date_str()
        assert(cv2.imwrite('pics/{}-motion-full.jpg'.format(dstr), current_image))
        #assert(cv2.imwrite('pics/{}-motion.jpg'.format(dstr), prior_image))
        #assert(cv2.imwrite('pics/{}-diff.jpg'.format(dstr), diff))
        #assert(cv2.imwrite('pics/{}-thresh.jpg'.format(dstr), thresh))
        #assert(cv2.imwrite('pics/{}-dilated.jpg'.format(dstr), thresh))

        cropped_image = current_image[ly:ly+lh, lx:lx+lw]
        #assert(cv2.imwrite('assets/unsorted_cropped/{}-motion-cropped.jpg'.format(dstr), cropped_image))

    prior_image = current_image
    return motion, cropped_image

def detect_cat_local(full_image, cropped_image):
    prediction = predict(cropped_image)

    # Set aside image for future training, with prediction in file name
    dstr = date_str()
    assert(cv2.imwrite('assets/unsorted_cropped/{}-{}.jpg'.format(prediction, dstr), cropped_image))

    cat_detected = prediction == "cat"
    if cat_detected:
        #assert(cv2.imwrite('pics/{}-cat.jpg'.format(date_str()), image))

        # Save image as current for the web server
        cpy = np.array(full_image)
        cv2.putText(cpy, '{:%Y-%m-%d %H:%M}'.format(datetime.now()), (4, 36),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 6)
        cv2.putText(cpy, '{:%Y-%m-%d %H:%M}'.format(datetime.now()), (4, 36),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        assert(cv2.imwrite('/var/www/html/pics/current.jpg', cpy))
    return cat_detected

try:
    while True:
        # Wait a bit
        camera.wait_recording(step_sec)

        # Grab a picture
        stream = io.BytesIO()
        camera.capture(stream, format='jpeg', use_video_port=True)
        stream.seek(0)
        raw_image = Image.open(stream)
        image = np.array(raw_image) 

        # Detect motion and then cat
        motion, cropped_image = detect_motion(image)
        if motion:
            print("Detected motion", date_str())
            if detect_cat_local(image, cropped_image):
                print("  Detected cat!", date_str())

finally:
    camera.stop_recording()
