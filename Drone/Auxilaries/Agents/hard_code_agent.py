import numpy as np
import tf_agents
from tf_agents.policies import py_policy


class HardCodeAgent(py_policy.PyPolicy):
	def __init__(self, time_step_spec, action_spec, goals = [64, 20], policy_state_spec=()):
		self._time_step_spec = time_step_spec
		self._action_spec = action_spec
		self._policy_state_spec = policy_state_spec
		self.goals = goals

	def _action(self, time_step, policy_state=()):
        # Find the bounding box pixels
		indexes = np.where(time_step.observation[0,:,:,0] == -1)
		
        # If its in view, get the outer box points
		if indexes[0].size != 0 or indexes[1].size != 0:
			bounding_box_begin = (min(indexes[0]), min(indexes[1]))
			bounding_box_end = (max(indexes[0]), max(indexes[1]))
			bounding_box = (bounding_box_begin, bounding_box_end)
			
        # Else, keep turning until you do find a bounding box
		else:
			return tf_agents.trajectories.policy_step.PolicyStep(action=np.array(1), state=(), info=())

        # Get the center of the bounding box
		center_box_x        = ((bounding_box[1][1] - bounding_box[0][1])/2) + bounding_box[0][1]

        # If it centered, check if the bounding box height is correct
		if center_box_x < self.goals[0]+(self.goals[0]*0.2) and center_box_x > self.goals[0]-(self.goals[0]*0.2):
			bb_height = abs(bounding_box[1][0] - bounding_box[0][0])
			if bb_height < self.goals[1]*0.2 and bb_height > (self.goals[1]-(self.goals[1]*0.2)):
				action =  0
			elif bb_height < (self.goals[1]-(self.goals[1]*0.2)):
				action =  3
			elif bb_height > self.goals[1]*0.2:
				action =  0

        # In the other two cases, center the person
		elif center_box_x > self.goals[0]+(self.goals[0]*0.2):
			action = 1
		elif center_box_x < self.goals[0]-(self.goals[0]*0.2):
			action = 2
		return tf_agents.trajectories.policy_step.PolicyStep(action=np.array(action), state=(), info=())


	def time_step_spec(self):
		return self._time_step_spec

	def action_spec(self):
		return self._action_spec

	def policy_state_spec(self):
		return self._policy_state_spec