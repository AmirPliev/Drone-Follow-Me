import subprocess
import airsim
import numpy as np
import cv2
import os
import json
import time

""" Open the environment on a specific port """
def openAirSim(port, display, multi, environment):
    filename = r"Auxilaries\Environments\UnrealEnvironments\{}\{}\Binaries\Win64\settings.json".format(environment, environment)

    # If we have multiple environments in parallel, we open multiple, each on different ports
    if multi == True:

        # Open our settings file
        with open(filename, "r") as jsonFile:
            data = json.load(jsonFile)

        # Change the port
        data["ApiServerPort"] += 1
        port = data["ApiServerPort"]

        # Change our ViewMode accordingly
        if display == False:
            data["ViewMode"] = "NoDisplay"
        else:
            data["ViewMode"] = "SpringArmChase"

        # Save the settings file
        with open(filename, "w") as jsonFile:
            json.dump(data, jsonFile)

        # Open AirSim with these settings
        subprocess.call([r'Auxilaries\Environments\UnrealEnvironments\{}\PythonRun.bat'.format(environment)], shell=True)

        # Connect to the AirSim Drone
        client = airsim.MultirotorClient(port = port)
        client.confirmConnection()
        airsim.DrivetrainType.ForwardOnly
        client.enableApiControl(True)
        client.armDisarm(True)
        client.takeoffAsync().join()

    # With a single environment we only change what is relevant. 
    else:
        if port != 41451:
            with open(filename, "r") as jsonFile:
                data = json.load(jsonFile)

            data["ApiServerPort"] = port

            with open(filename, "w") as jsonFile:
                json.dump(data, jsonFile)

        if display == False:
            with open(filename, "r") as jsonFile:
                data = json.load(jsonFile)

            data["ViewMode"] = "NoDisplay"

            with open(filename, "w") as jsonFile:
                json.dump(data, jsonFile)
        
        # Start AirSim with given resolution
        subprocess.call([r'Auxilaries\Environments\UnrealEnvironments\{}\PythonRun.bat'.format(environment)], shell=True)
       
        # Connect to the AirSim Drone
        client = airsim.MultirotorClient(port = port)
        client.confirmConnection()
        airsim.DrivetrainType.ForwardOnly
        client.enableApiControl(True)
        client.armDisarm(True)
        client.takeoffAsync().join()

        # Change it back to the default value
        if port != 41451:
            with open(filename, "r") as jsonFile:
                data = json.load(jsonFile)

            data["ApiServerPort"] = 41451

            with open(filename, "w") as jsonFile:
                json.dump(data, jsonFile)

        if display == False:
            with open(filename, "r") as jsonFile:
                data = json.load(jsonFile)

            data["ViewMode"] = "SpringArmChase"

            with open(filename, "w") as jsonFile:
                json.dump(data, jsonFile)
    return client
        
""" Draw the bounding for Debug mode """
def drawBoundingBox(image, bb_begin, bb_end, method = "showImageOnly"):

    if method == "showBoundingBox":
        cv2.line(image, bb_begin, (bb_begin[0], bb_end[1]), (0,255,0), thickness=2)
        cv2.line(image, (bb_begin[0], bb_end[1]), bb_end, (0,255,0), thickness=2)
        cv2.line(image, (bb_end[0], bb_begin[1]), bb_end, (0,255,0), thickness=2)
        cv2.line(image, bb_begin, (bb_end[0], bb_begin[1]), (0,255,0), thickness=2)
    
    cv2.imshow("image", image)
    cv2.waitKey()

""" Gets a segmentation map and creates a bounding box """    
def getBoundingBoxfromSegment(image):

    # Get the pixels with target and create bounding box points (if not empty)
    target_indexes = np.where((image[:,:,0]==73) & (image[:,:,1]==37) & (image[:,:,2]==226))
    if target_indexes[0].size != 0 or target_indexes[1].size != 0:
        bounding_box_begin  = (min(target_indexes[0]), min(target_indexes[1]))
        bounding_box_end    = (max(target_indexes[0]), max(target_indexes[1]))
        return bounding_box_begin, bounding_box_end
    else:
        return None, None

""" Request the images from AirSim """
def getImagesfromSim(client, depth_imaging = False):

    if depth_imaging == True:

        # Get the depth image and segmentation map
        responses = client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.DepthVis, True, False), 
                                                airsim.ImageRequest(0, airsim.ImageType.Segmentation, False, False)])

        # Transform both to numpy
        depth               = responses[0]
        depth_image         = airsim.list_to_2d_float_array(depth.image_data_float, depth.width, depth.height) 

        depth_image         = np.clip(depth_image, 0, 100)
        depth_image         = np.divide(depth_image, 100, dtype=np.float32)

        segment             = np.fromstring(responses[1].image_data_uint8, dtype=np.uint8) 
        segment_img_rgb     = segment.reshape(responses[1].height, responses[0].width, 3) 

        return depth_image, segment_img_rgb

    elif depth_imaging == False:
            # Get the depth image and segmentation map
        responses = client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.Scene, False, False),  
                                                airsim.ImageRequest(0, airsim.ImageType.Segmentation, False, False)])

        # Process image to grayscale and normalized
        response = responses[0]
        img1d       = np.fromstring(response.image_data_uint8, dtype=np.uint8) 
        img_rgb     = img1d.reshape(response.height, response.width, 3)
        img_gray    = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        img_gray    = np.divide(img_gray, 255, dtype=np.float32)

        # Process segment map
        segment             = np.fromstring(responses[1].image_data_uint8, dtype=np.uint8) 
        segment_img_rgb     = segment.reshape(responses[1].height, responses[0].width, 3) 

        return img_gray, segment_img_rgb

""" Process a single image and boundingbox """
def getSingleImage(client, depth_imaging, BBmethod = "segment"):
    image, segment_img_rgb = getImagesfromSim(client, depth_imaging = depth_imaging)

    # Get the bounding box according to method
    if BBmethod == "segment":
        bounding_box_begin, bounding_box_end = getBoundingBoxfromSegment(segment_img_rgb)
    elif BBmethod == "yolo":
        pass
    
    # Set the pixels to negative values
    if bounding_box_begin is not None or bounding_box_end is not None:
        image[bounding_box_begin[0]:bounding_box_end[0], bounding_box_begin[1]: bounding_box_end[1]]= -1

    return image, (bounding_box_begin, bounding_box_end)

""" Save the collisions """
def writeCollision(collisionInfo, name):
    if name == 'hard' or name == 'random' or name == "DefaultName":
        filename = r"Auxilaries\TrainedModels\{} - Collisions".format(name)
    else:
        filename = r"Auxilaries\TrainedModels\{}\{} - Collisions".format(name, name)
    f=open(filename,'a')
    np.savetxt(f, collisionInfo)
    f.close()

""" Save the Reward Distributions """
def writeRewardDistribution(name, array, episode_count):
    if name == 'hard' or name == 'random' or name == "DefaultName":
        filename = r"Auxilaries\TrainedModels\{} - RewardDistribution".format(name)
    else:
        filename = r"Auxilaries\TrainedModels\{}\{} - RewardDistribution".format(name, name)
    np.savetxt(filename, np.append(array, episode_count))

""" Save the paths """
def writePath(positionInfo, name, end):
    if name == 'hard' or name == 'random' or name == "DefaultName":
        filename = r"Auxilaries\TrainedModels\{} - Paths".format(name)
    else:
        filename = r"Auxilaries\TrainedModels\{}\{} - Paths".format(name, name)
    f=open(filename,'a')
    np.savetxt(f, positionInfo)
    f.close()

    if end:
        with open(filename, 'a') as file:
            file.write('---\n')

""" Save the OutofViews """
def writeOutOfView(position, name):
    if name == 'hard' or name == 'random' or name == "DefaultName":
        filename = r"Auxilaries\TrainedModels\{} - OutOfView".format(name)
    else:
        filename = r"Auxilaries\TrainedModels\{}\{} - OutOfView".format(name, name)
    f=open(filename,'a')
    np.savetxt(f, position)
    f.close()
    