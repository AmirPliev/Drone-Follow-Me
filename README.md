# Drone Follow Me @ML6 #
Drone - Computer Vision - Machine Learning - Artificial Intelligence



## Goal ##
Research and create a Machine Learning algorithm that can follow a person or vehicle in 
images captured by a drone operation on a computationally limited device. 


## Challenge ##

Most importantly is the object detection system. In order to accurately track a person, object 
detection algorithms need to be used in order to estimate where the follow-target is. Using this input, 
the task becomes now how to decide what action to take. Instead of hard-coding every situation with the 
most appropriate action, the challenge would be to see whether the drone could teach itself the most 
appropriate behavior. This should all be able to run on an embedded device, in this case the Raspberry Pi 4B. 


## Technique ##

### YOLO ###
A popular algorithm nowadays in the computer vision world is YOLO (You Only Look Once), this algorithm 
proved to be state-of-the-art in real-time object detection. However, using YOLO on a device that is 
computationally limited can form problems with the processing of the incoming stream of images. Therefore, 
smaller variants of the YOLO algorithm will be used, with the primary focus being YOLO-Fastest. 

### Depth Perception ###
Seeing obstacles is a challenge in itself, but it is especially important on an embedded device such as 
the Raspberry Pi. A possible solution is to create depth-images using either a depth-perception camera. 
This could be then used for the action-decision problem. 

### Reinforcement Learning ###
Furthermore, the use of Reinforcement Learning (RL) will be used in order to let the drone teach itself 
how to react to each situation. This will allow the agent to generalize to new situations that have not 
been predicted in advance. Since training RL algorithms are very data-hungry, the challenge will be to 
first train the agent in a simulator. In this case, Microsoft's Airsim will be used. The trained agent 
will then be transferred to the physical drone in order to test its performance. 


### Who do I talk to? ###

* Amir Pliev
* Laurens Weijs