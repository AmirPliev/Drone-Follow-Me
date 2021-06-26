from picamera import PiCamera
from picamera.array import PiRGBArray
import Yolo_Fastest
import Yolo_Fastest.darknet_images as detection
import Yolo_Fastest.darknet as dn
from time import sleep
import cv2
import time
import random


# Instantiate camera
camera = PiCamera()
camera.resolution = (256, 144)
camera.framerate = 32

rawCapture = PiRGBArray(camera, size = (256, 144))
time.sleep(0.1)
random.seed(3)  # deterministic bbox colors

# Load the YOLO network
network, class_names, class_colors = dn.load_network(
            'Yolo_Fastest/ModelZoo/yolo-fastest-1.1_body/yolo-fastest-1.1_body.cfg',
            'Yolo_Fastest/ModelZoo/yolo-fastest-1.1_body/body.data',
            'Yolo_Fastest/ModelZoo/yolo-fastest-1.1_body/yolo-fastest-1.1_body.weights',
            batch_size=1)


for x in range(10):
        
    # make a picture
    camera.capture(rawCapture, format = 'bgr')
    image = rawCapture.array
    
    # Make a prediction and measure the time    
    prev_time = time.time()
    image, detections = detection.image_detection(
        image, network, class_names, class_colors, 0.25)
    fps = int(1/(time.time() - prev_time))
    print("FPS: {}".format(fps))

    # Resize the image for TF-Agents
    image = cv2.resize(image, (512,288), interpolation=cv2.INTER_LINEAR)
        
    if len(detections) == 0:
        print("WHOOPS, NO PERSON FOUND")
        continue

    bounding_box = dn.bbox2points(detections[0][2])
    #print(bounding_box)

    image_center = 128
    height_goal  = int(256/3)

    # Get the center of the bounding box
    center_box_x        = ((bounding_box[2] - bounding_box[0])/2) + bounding_box[0]
    bb_height = abs(bounding_box[3] - bounding_box[1])
    
    # If it centered, check if the bounding box height is correct
    if center_box_x < image_center+(image_center*0.2) and center_box_x > image_center-(image_center*0.2):
        
        if bb_height < height_goal*0.2 and bb_height > (height_goal-(height_goal*0.2)):
            action =  0
        elif bb_height < (height_goal-(height_goal*0.2)):
            action =  3
        elif bb_height > height_goal*0.2:
            action =  0
        

    # In the other two cases, center the person
    elif center_box_x > image_center+(image_center*0.2):
        action = 1
    elif center_box_x < image_center-(image_center*0.2):
        action = 2
        
    print("Height: ", bb_height)
    
    thick = 2
    color = (0,0,255)

        
    if action == 0:
        print("DO NOTHING")
    elif action == 1:
        print("ROTATE RIGHT")
        cv2.arrowedLine(image, (10, 10), (80, 10), color, thick)
    elif action == 2:
        print("ROTATE LEFT")
        cv2.arrowedLine(image, (80, 10), (10, 10), color, thick)
    elif action == 3:
        print("MOVE STRAIGHT")
        cv2.arrowedLine(image, (45, 80), (45, 10), color, thick)
        
    cv2.imwrite("Demo_Results/image_" + str(x) + ".jpg", image)
        
    print(" ")
    rawCapture.truncate(0)
