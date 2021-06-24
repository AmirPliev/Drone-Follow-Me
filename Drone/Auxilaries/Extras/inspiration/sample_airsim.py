# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 07:56:18 2018

@author: woill
"""

from pyquaternion import Quaternion
import numpy as np

import matplotlib.pyplot as plt
plt.ion()

def  init_client(AirSim =True , UnrealCV =True , duration = 10):
    clients = []

    if UnrealCV:
        from unrealcv import client as client_cv
        from unrealcv.automation import UE4Binary
        from unrealcv.util import read_png, read_npy
        client_cv.connect()
        clients.append(client_cv)
    if AirSim:
        import os
        import tempfile
        import pprint
        client = MultirotorClient()
        client.confirmConnection()
        client.enableApiControl(True)
        client.armDisarm(True)
        #client.takeoff()
        #client.hover()
        client.moveByVelocity(0,0,-20, duration = duration ,  drivetrain = DrivetrainType.ForwardOnly , yaw_mode = YawMode(is_rate = False , yaw_or_rate =0.0) )
        clients.append(client)

    return clients

def show_current_frame(client): #UnrealCV function
    res = client.request('vget /camera/0/lit png'); img = read_png(res); plt.imshow(img)



def run_a_pov(client , actions = [-2, -4,-1, 45] , duration = 2 , drivetrain = DrivetrainType.MaxDegreeOfFreedom):
    #actions =  +North, +East,+Down, Angle (from -180 to +180) in degree/sec
    action = actions[0:3]
    q = client.getCameraInfo(0).pose.orientation #this is different from the getOrientation, taking the latter can create problem due to the fact that the drone move slower than the cam?
    my_quaternion = Quaternion(w_val=q.w_val,x_val=q.x_val,y_val= q.y_val,z_val=q.z_val)
    mvm = my_quaternion.rotate(action)
    donre_vel_rota = [client.getVelocity().x_val , client.getVelocity().y_val , client.getVelocity().z_val]
    client.moveByVelocity(vx = donre_vel_rota[0] + mvm[0], #the already existing speed + the one the agent wants to add, smoother drive?
                          vy = donre_vel_rota[1] + mvm[1],
                          vz = donre_vel_rota[2] + mvm[2],
                          duration = duration , #will last x secondes or will be stoped by a new command (put a time.sleep(0.5) next to it)
                          drivetrain =  drivetrain, #the camera is indepedant of the movement, but the movement is w.r.t the cam orientation
                          yaw_mode = YawMode(is_rate = True, yaw_or_rate = actions[3]) ) # True means that yaw_or_rate is seen as a degrees/sec


client = init_client(True, False , duration = 4)[0]

#actions = [0, -2,-0.8, 180] # +North, +East,+Down, Angle
for i in range (100):
    actions = [0.5, 0.3, -0.2, 0]
    run_a_pov(client , actions = actions)
    time.sleep(0.1)