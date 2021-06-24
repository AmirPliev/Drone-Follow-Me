import airsim
import numpy as np
import cv2
import math
import time
import tensorflow as tf
from pyquaternion import Quaternion
import json
import random
import psutil

#tf-agents
import tf_agents.environments as Environments
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

# My own modules
import Auxilaries.Environments.Additionals.RewardFunctions as Rewards
import Auxilaries.Environments.Additionals.Movements as Movements
import Auxilaries.Environments.Additionals.AirSimHelpers as AirSimHelpers


class Environment(Environments.py_environment.PyEnvironment):
	def __init__(self,  reward_fn,
						reset_method,
						target          = "rp_carla_rigged_001_Mobile_ue4_5", 
						port            = 41451, 
						display         = True, 
						multi           = True, 
						continuous      = False,
						depth_on        = False,
						stack_on        = False,
						environment		= "BlocksNormal",
						evaluation		= 0,
						name 			= "DefaultName"
						):

		# Open airsim, unles evaluation mode = 4, then we just connect directly
		if evaluation != 4:
			self.client         = AirSimHelpers.openAirSim(port, display, multi, environment)
		else:
			client = airsim.MultirotorClient()
			client.confirmConnection()
			airsim.DrivetrainType.ForwardOnly
			client.enableApiControl(True)
			client.armDisarm(True)
			client.takeoffAsync().join()

		self.depth_on       = depth_on
		self.stack_on       = stack_on
		self.continuous     = continuous
		self.target_name    = target
		self.reward_fn      = reward_fn
		self.reset_method   = reset_method
		self.evaluation 	= evaluation
		self.environment	= environment
		self.name			= name

		# Set the color of our target object in the segmentation map so we can find it later in our segmentation maps
		success = self.client.simSetSegmentationObjectID(target, 12, True)
		print("Setting ID for Segmentation Map: ", success)

		# Get state
		self._state, box  = self.getState(depth_imaging = self.depth_on, stacked_imaging = self.stack_on)

		# Set the tensor specs for actions and observations
		if continuous:
			self._action_spec = array_spec.BoundedArraySpec(shape=(3,), dtype=np.float32, minimum=0, maximum=1, name = 'action')
		else:
			self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=5, name = 'action')

		if stack_on:
			self._observation_spec = array_spec.BoundedArraySpec(
				shape=(self._state.shape[0], self._state.shape[1], self._state.shape[2]), dtype=np.float32, minimum=-1, maximum=1, name='observation')
		else:
			self._observation_spec = array_spec.BoundedArraySpec(
				shape=(self._state.shape[0], self._state.shape[1], 1), dtype=np.float32, minimum=-1, maximum=1, name='observation')

		# Set overall environment variables
		self._episode_ended             = False
		self.z                          = -2
		self.reward_distribution		= np.zeros(50)
		self.episode_count				= 0
		self.reset()
		self.timestamp                  = 0
		self.time_end                   = False
		self.steps                      = 0
		self.beginning                  = True
		self.steps_out_of_view          = 0
		self.pose                       = self.client.simGetVehiclePose()
		self.pose.position.z_val        = self.z
		self.current_bounding_box       = box
		self.image_center               = (self._state.shape[0] / 2, self._state.shape[1] / 2)
		   
	""" OVERRIDDEN: Reset the drone to the original state """
	def _reset(self):

		# Reset the drone to the origin position
		if self.reset_method == "OriginalPlace":
			self.client.reset()
			self.client.simSetVehiclePose(self.pose, ignore_collison=False)
			self.client.enableApiControl(True)
			self.client.armDisarm(True)
			self.client.takeoffAsync()
			self.client.hoverAsync().join()

		# Reset the drone to a random position around the person
		elif self.reset_method == "RandomPlaceAround":
			self.client.reset()
			self.client.simSetVehiclePose(self.pose, ignore_collison=False)

			# Get coordinates of drone and person
			pose                = self.client.simGetVehiclePose()
			person              = self.client.simGetObjectPose(self.target_name)

			# Set a new position
			pose.position.x_val = person.position.x_val + ([-1,1][random.randrange(2)] * random.uniform(5, 10))
			pose.position.y_val = person.position.y_val + ([-1,1][random.randrange(2)] * random.uniform(5, 10))
			pose.position.z_val = self.z

			# Calculate new angle to look at 
			differences         = [person.position.x_val - pose.position.x_val, person.position.y_val - pose.position.y_val]
			new_quat            = np.array(airsim.to_eularian_angles(pose.orientation))
			new_quat[2]         = np.arctan((differences[1]/differences[0]))

			# Correction for inversed position
			if differences[0] < 0:
				new_quat[2]         += math.pi
			
			# Move the drone to position and rotation
			pose.orientation    = airsim.to_quaternion(new_quat[0], new_quat[1], new_quat[2])
			self.client.simSetVehiclePose(pose, ignore_collison=False)
			self.client.enableApiControl(True)
			self.client.armDisarm(True)
			self.client.takeoffAsync()
			self.client.hoverAsync().join()

		# Reset the drone directly behind the person facing the person
		elif self.reset_method == "DirectlyBehind":

			# Get coordinates of drone and person
			pose                = self.client.simGetVehiclePose()
			person              = self.client.simGetObjectPose(self.target_name)

			# Correct the Eulerian angle and calculate the new position
			angle               = airsim.to_eularian_angles(person.orientation)[2]
			if angle < 0:
				angle += math.pi
			else:
				angle = angle-math.pi
			change_x			= 4 * np.sin(angle)
			change_y			= 4 * np.cos(angle)

			# Set the new position
			pose.position.x_val = person.position.x_val - change_x
			pose.position.y_val = person.position.y_val + change_y
			pose.position.z_val = self.z

			# Calculate new angle to look at 
			differences         = [person.position.x_val - pose.position.x_val, person.position.y_val - pose.position.y_val]
			new_quat            = np.array(airsim.to_eularian_angles(pose.orientation))
			new_quat[2]         = np.arctan((differences[1]/differences[0]))

			# Correction for inversed position
			if differences[0] < 0:
				new_quat[2]         += math.pi
			
			# Move the drone to position and rotation and reconnect
			pose.orientation    = airsim.to_quaternion(new_quat[0], new_quat[1], new_quat[2])
			self.client.simSetVehiclePose(pose, ignore_collison=True)
			self.client.enableApiControl(True)
			self.client.armDisarm(True)
			self.client.takeoffAsync()
			self.client.hoverAsync().join()

		# Reset the environment variables 
		self._episode_ended     = False 
		self.steps              = 0
		self.beginning          = True
		self.steps_out_of_view  = 0
		self.episode_count		+= 1

		# Write the reward distributions
		if self.evaluation != 0 and self.evaluation != 2:
			AirSimHelpers.writeRewardDistribution(self.name, self.reward_distribution, self.episode_count)

		# Return final timestep
		return ts.restart(np.array(self._state, dtype=np.float32)) 

	""" Just reset position, without affecting the episode """
	def ensureHeight(self):
		height = self.client.simGetVehiclePose().position.z_val
		pose = self.client.simGetObjectPose(self.target_name)
		if height > self.z + 0.3 or height < self.z - 0.3:
			pose.position.z_val = self.z
			self.client.simSetVehiclePose(self.pose, ignore_collison=False)
			self.client.enableApiControl(True)
			self.client.armDisarm(True)
			self.client.takeoffAsync()
			self.client.hoverAsync().join()

	""" OVERRIDDEN: Perform an action, get a new state and calculate reward """
	def _step(self, action):

		# Make sure the drone is at the correct height
		self.ensureHeight()

		# If the episode is done, reset the environment
		if self._episode_ended:
			return self.reset()

		# Perform the chosen move and see if there was a collision (if so, end the episode)
		self._episode_ended = self.move(action = action, continuous = self.continuous)

		# Get the new state
		self._state, bounding_box = self.getState(depth_imaging=self.depth_on, stacked_imaging=self.stack_on)
		self.current_bounding_box = bounding_box

		# Determine reward
		reward = self.reward_fn(bounding_box, self.image_center)
		if self.evaluation != 0 and self.evaluation != 2:
			self.reward_distribution[self.steps] 	+= reward

		# Count steps
		self.steps += 1

		# Check whether this is the first time that the person is out of view and keep track
		if bounding_box[0] is None or bounding_box[1] is None:
			self.steps_out_of_view += 1
		else:
			self.steps_out_of_view = 0
			
		# Call episode end if person is out of view for too long or reach our limit
		if  (self.steps_out_of_view >= 1 or self.steps >= 50) and self.evaluation != 2:
			self._episode_ended = True
			self.time_end = True

		if self.evaluation != 0:
			AirSimHelpers.writePath([self.client.simGetVehiclePose().position.x_val, self.client.simGetVehiclePose().position.y_val], self.name, end = self._episode_ended)

		# If we're at the end of an episode, which means we've hit a collision, give a horrible reward and terminate
		if self._episode_ended:

			# If termination was not due to timeout, give corresponding reward
			if self.time_end == False or self.steps_out_of_view >= 1:
				reward = -1
				if self.evaluation != 0:
					AirSimHelpers.writeOutOfView([self.client.simGetVehiclePose().position.x_val, self.client.simGetVehiclePose().position.y_val], self.name)

			return ts.termination(np.array(self._state, dtype=np.float32), reward)

		# If were still going on we transition to the next state
		else:
			return ts.transition(np.array(self._state, dtype=np.float32), reward=reward, discount=0.9)      

	""" Take a move in the world """
	def move(self, action, duration = 0.1, continuous = False):

		# Discrete actions
		if continuous == False:
			self.client.hoverAsync()

			if self.client.simGetCollisionInfo().has_collided:
				if self.evaluation != 0:
					AirSimHelpers.writeCollision([self.client.simGetCollisionInfo().impact_point.x_val, self.client.simGetCollisionInfo().impact_point.y_val], self.name)
				return True

			# Do nothing
			if action == 0:
				self.client.moveByVelocityAsync(0, 0, 0, 1)
				self.client.rotateByYawRateAsync(0, 1)
				if self.client.simGetCollisionInfo().has_collided:
					return True
			
			# Orient Right
			if action == 1:
				Movements.yaw_right(self.client, 50, 0.1)
				if self.client.simGetCollisionInfo().has_collided:
					return True

			# Orient Left
			if action == 2:
				Movements.yaw_left(self.client, 50, 0.1)
				if self.client.simGetCollisionInfo().has_collided:
					return True

			# Go straight
			if action == 3:
				Movements.straight(self.client, 6, duration, "straight", self.z)
				if self.client.simGetCollisionInfo().has_collided:
					return True

			# Go right
			if action == 4:
				Movements.straight(self.client, 6, duration, "right", self.z)
				if self.client.simGetCollisionInfo().has_collided:
					return True

			# Go left
			if action == 5:
				Movements.straight(self.client, 6, duration, "left", self.z)
				if self.client.simGetCollisionInfo().has_collided:
					return True

			return False
		
		# Continuous Movements
		else:
			if self.client.simGetCollisionInfo().has_collided:
				return True
			Movements.continuousMove(self.client, action, duration, self.z)
			if self.client.simGetCollisionInfo().has_collided:
					return True

	""" Get the state according to the preferred method """
	def getState(self, depth_imaging, stacked_imaging, BBmethod = "segment", image_set_size = 3):

		# Single images
		if not stacked_imaging:
			time.sleep(0.4)
			image, bounding_box = AirSimHelpers.getSingleImage(self.client, depth_imaging=self.depth_on, BBmethod=BBmethod)
			return np.expand_dims(image, axis=2), bounding_box
		
		# Stacked images
		else:
			interval = 0.1
			image_set = []

			for _ in range(image_set_size):
				image, bounding_box = AirSimHelpers.getSingleImage(self.client, depth_imaging=self.depth_on, BBmethod=BBmethod)
				
				# Collect the image and wait 
				image_set.append(image)
				time.sleep(interval)

			# Stack images into one object and return latest bounding box
			images = np.stack(image_set, axis = 2)
			return images, bounding_box

	def action_spec(self):
		return self._action_spec

	def observation_spec(self):
		return self._observation_spec

	def setPause(self):
		self.client.simPause(True)

	def unPause(self):
		self.client.simPause(False)

	def Kill(self, processName):
		#Iterate over the all the running process
		for proc in psutil.process_iter():
			try:
				# Check if process name contains the given name string.
				if processName.lower() in proc.name().lower():
					proc.kill()
					return True
			except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
				pass
		return False;