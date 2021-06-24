from tf_agents.utils import common
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import q_network
from tf_agents.networks.actor_distribution_network import ActorDistributionNetwork
from tf_agents.agents.ppo.ppo_clip_agent import PPOClipAgent
from tf_agents.agents.ppo import ppo_agent
from tf_agents.networks.value_network import ValueNetwork
import tf_agents

import random
import numpy as np
import tensorflow as tf


""" Just a definition to create the agent """
def getAgent(environment, learning_rate, global_step, which = "ppo"):

	fc_layer_params = (512,128)
	conv_layer_params = ((32, 8, 4), (64, 4, 2), (64, 3, 1))

	if which == "dqn":

		# Network 
		q_net = q_network.QNetwork(
			environment.observation_spec(),
			environment.action_spec(),
			conv_layer_params   = conv_layer_params,
			fc_layer_params     = fc_layer_params,
			batch_squash        = True)
			
		"""
		q_net = sequential.Sequential([
			tf.keras.layers.Conv2D(16, 3, padding= 'same' , activation='relu'),
			tf.keras.layers.
    	])
		"""
		
		optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=learning_rate )


		agent = dqn_agent.DqnAgent(
			environment.time_step_spec(),
			environment.action_spec(),
			q_network           = q_net,
			optimizer           = optimizer,
			td_errors_loss_fn   = common.element_wise_squared_loss,
			train_step_counter  = global_step)

		return agent

		
	elif which == "ppo":
		actor_net, value_net = create_networks(environment.observation_spec(), environment.action_spec(), fc_layer_params, conv_layer_params)

		optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=learning_rate)
		agent = PPOClipAgent(
			time_step_spec 		        = environment.time_step_spec(),
			action_spec 		        = environment.action_spec(),
			optimizer			        = optimizer,
			actor_net 			        = actor_net,
			value_net			        = value_net,
			train_step_counter	        = global_step,
            use_gae                     = True,
            importance_ratio_clipping   = 0.2,
            discount_factor             = 0.99
		)
		return agent

""" Make the networks for our PPO agent """
def create_networks(observation_spec, action_spec, fc_layer_params, conv_layer_params):

	actor_net = ActorDistributionNetwork(
		observation_spec,
		action_spec,
		fc_layer_params     = fc_layer_params,
		conv_layer_params   = conv_layer_params
		)
	value_net = ValueNetwork(
		observation_spec,
		fc_layer_params     = fc_layer_params,
		conv_layer_params   = conv_layer_params
		)

	return actor_net, value_net


