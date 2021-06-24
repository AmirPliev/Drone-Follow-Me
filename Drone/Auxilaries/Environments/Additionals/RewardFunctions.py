import numpy as np

# Very simple and sparse reward function that looks at whether person is centered, close to goal height and above certain horizon
def SimpleSparseClose(boundingbox, image_center):

    # If person is not in view, give a negative reward
    if boundingbox[0] is None or boundingbox[1] is None:
        return -1
    
    # Calculate the center of the bounding box, the permitted margin and the actual distance of the BB to the center of the image
    center_box_x        = ((boundingbox[1][1] - boundingbox[0][1])/2) + boundingbox[0][1] 
    center_cutoff       = image_center[1]*0.2
    distance_to_centre  = abs(center_box_x - image_center[1])
    
    # Calculate the BB height, position of the centre of the BB and the proportion in which this value can fall
    bb_height           = abs(boundingbox[1][0] - boundingbox[0][0])
    bb_center_position  = ((image_center[0] * 2)  - boundingbox[1][0]) + ((abs(boundingbox[1][0] - boundingbox[0][0]))/2)
    proportion          = (bb_center_position / (image_center[0] * 2) )*100

    # Give reward according to the following conditions:
    if distance_to_centre < center_cutoff and bb_height <= 30 and bb_height >= 18  and proportion > 17:
        return 1
    else:
        return 0

# Same reward as SimpleSparseClose but with tighter margins
def SimpleSparseCloseLessMargin(boundingbox, image_center):

    # If person is not in view, give a negative reward
    if boundingbox[0] is None or boundingbox[1] is None:
        return -1
    
    # Calculate the center of the bounding box, the permitted margin and the actual distance of the BB to the center of the image
    center_box_x        = ((boundingbox[1][1] - boundingbox[0][1])/2) + boundingbox[0][1] 
    center_cutoff       = image_center[1]*0.2
    distance_to_centre  = abs(center_box_x - image_center[1])
    
    # Calculate the BB height, position of the centre of the BB and the proportion in which this value can fall
    bb_height           = abs(boundingbox[1][0] - boundingbox[0][0])
    bb_center_position  = ((image_center[0] * 2)  - boundingbox[1][0]) + ((abs(boundingbox[1][0] - boundingbox[0][0]))/2)
    proportion          = (bb_center_position / (image_center[0] * 2) )*100

    # Give reward according to the following conditions, this time margins are tighter:
    if distance_to_centre < center_cutoff and bb_height <= 28 and bb_height >= 24  and proportion > 17:
        return 1
    else:
        return 0
    

""" Calculate reward based on center of bounding box and height """
def BoundingBoxAndCenter(boundingbox, image_center,  weight_height = 0.8, bounding_box_ideal_height = 24):

    weight_center = 1 - weight_height

    # If person is not in view, give a negative reward
    if boundingbox[0] is None or boundingbox[1] is None:
        return 0 

    else:
        # Weight ascribed to centering the person in the image
        center_box_x        = ((boundingbox[1][1] - boundingbox[0][1])/2) + boundingbox[0][1] 
        center_cutoff       = image_center[1]*0.2
        distance_to_centre  = abs(center_box_x - image_center[1]) - center_cutoff
        if distance_to_centre <= center_cutoff:
            reward1         = weight_center
        else:
            reward1         = weight_center - (distance_to_centre / ((image_center[1]-center_cutoff) / weight_center))
        
        # Weight ascribed to the height of the bounding box | aka - distance to person
        bb_height           = abs(boundingbox[1][0] - boundingbox[0][0])
        bb_goal_cutoff      = bounding_box_ideal_height - (bounding_box_ideal_height * 0.2)

        if bb_height >= bb_goal_cutoff and bb_height <= bounding_box_ideal_height:
            reward2         = weight_height
        else:
            reward2             =  (((-1 * weight_height) * abs(bb_height - bb_goal_cutoff)) / (bb_goal_cutoff)) + weight_height
        if reward2 < 0:
            reward2 = 0
    
        reward = reward1 + reward2
        assert reward <= 1 and reward >= 0

        return reward

""" Gets you the right reward during load time """
def getReward(reward):
    if reward == "SimpleSparseClose":
        return SimpleSparseClose
    elif reward == "SimpleSparseCloseLessMargin":
        return SimpleSparseCloseLessMargin
    elif reward == "BoundingBoxAndCenter":
        return BoundingBoxAndCenter

