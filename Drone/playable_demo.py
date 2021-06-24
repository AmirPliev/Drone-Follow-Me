import numpy as np
import time
import tensorflow as tf
import os
import subprocess
import keyboard

# My modules
from Auxilaries.Extras import AdditionalMethods as extras
from Auxilaries.Environments.Environment import Environment
import Auxilaries.Extras.Parameters as Parameters
import Auxilaries.Environments.Additionals.RewardFunctions as Rewards


""" Open environment """ 
env = Environment(  reward_fn       = Rewards.BoundingBoxHeight,
                    reset_method    = Parameters.Reset_Method.Directly_Behind_Person,
                    stack_on        = True,
                    depth_on        = True,
                    environment     = "BlocksNormal",
                    evaluation      = 1,
                    #target          = "rp_manuel_rigged_001_Mobile_ue4_5"
                )


time_step = env.reset()
rewards = []
counter = 0


while counter >= 0: 
	if keyboard.is_pressed('q'):
		time_step = env.step(2)
	elif keyboard.is_pressed('e'):
		time_step = env.step(1)
	elif keyboard.is_pressed('w'):
		time_step = env.step(3)
	elif keyboard.is_pressed('d'):
		time_step = env.step(4)
	elif keyboard.is_pressed('a'):
		time_step = env.step(5)
	elif keyboard.is_pressed('escape'):
		break
	else:
		time_step = env.step(0)
	counter += 1

	#print(time_step.reward)



