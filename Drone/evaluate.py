import tensorflow as tf 
from tf_agents.environments import tf_py_environment
from Auxilaries.Environments.Environment import Environment
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.drivers import dynamic_step_driver
import Auxilaries.Environments.Additionals.RewardFunctions as Rewards
import Auxilaries.Extras.Parameters as Parameters
from Auxilaries.Agents.hard_code_agent import HardCodeAgent
import Auxilaries.Extras.AdditionalMethods as Extras
from tf_agents.policies import random_tf_policy
from tf_agents.metrics import tf_metrics
import json
import numpy as np
import sys
import os
import time

def evaluate(EXPERIMENT = Parameters.Agent_Type.HARD_Agent, episodes = 10):
    with tf.device('/CPU:0'):


        """ Chooser menu for which model to evaluate """ 
        print("Available Models to Evaluate:")
        models = np.concatenate((["hard", "random"], os.listdir("Auxilaries/TrainedModels")))
        for x in range(len(models)):
            print(x, models[x])

        model_found = False
        while not model_found:
            try:
                model = int(input("Copy the model you want to evaluate: "))
                model_found = True
            except:
                print("You can't use this try again....")

        if model != '':
            EXPERIMENT = models[model]
        else:
            EXPERIMENT = 'hard'



        """ Load in the required parameters to use for evaluation """ 
        parameters = {      'Reward Function'   : Rewards.SimpleSparseClose,
                            'Reset Method'      : Parameters.Reset_Method.Directly_Behind_Person,
                            'Stacked Input'     : True,
                            'Depth Input'       : True }

        if EXPERIMENT != 'hard' and EXPERIMENT != 'random':
            parameters = Extras.transformToDict('Auxilaries/TrainedModels/' + EXPERIMENT +'/parameters.txt')
            parameters['Reward Function'] = Rewards.getReward(parameters['Reward Function'])
            print(parameters)



        """ Open environment """ 
        env = Environment(  reward_fn       = parameters['Reward Function'],
                            reset_method    = parameters['Reset Method'] ,
                            stack_on        = parameters['Stacked Input'] == "True",
                            depth_on        = parameters['Depth Input'] == "True",
                            environment     = "Factory",
                            evaluation      = 1,
                            name            = EXPERIMENT,
                            target          = "rp_manuel_rigged_001_Mobile_ue4_2"
                        )
        environment = tf_py_environment.TFPyEnvironment(env)



        """ Load the corresponding model """ 
        if EXPERIMENT == 'hard':
            policy = HardCodeAgent(environment.time_step_spec(), environment.action_spec())
        elif EXPERIMENT == 'random':
            policy = random_tf_policy.RandomTFPolicy(action_spec=environment.action_spec(), time_step_spec=environment.time_step_spec())
        else:
            policy_dir  = 'Auxilaries/TrainedModels/' + EXPERIMENT +'/policies'
            policy      = tf.compat.v2.saved_model.load(policy_dir)



        """ Perform the evaluation """ 
        if env.evaluation != 2:
            average_return            = tf_metrics.AverageReturnMetric()
            driver = dynamic_episode_driver.DynamicEpisodeDriver(environment, 
                                                            policy, 
                                                            [average_return], 
                                                            num_episodes= int(episodes))
            start = time.time()
            driver.run()
            print("Total: ", average_return.result())
            print("Performed in:   ", time.time() - start, " seconds")
        else:
            average_return = []
            for x in range(int(episodes)):
                time_step = environment.current_time_step()
                action_step = policy.action(time_step)
                next_time_step = environment.step(action_step.action)
                #print("Ep reward: ", int(next_time_step.reward))
                average_return.append(int(next_time_step.reward))
            print("Total: ", np.sum(average_return))
            print("Avg: ", np.mean(average_return))

        """ Kill """ 
        env.Kill("Blocks")
            


if __name__ == '__main__':

    if len(sys.argv) > 1:
        evaluate(episodes = sys.argv[1])
    else:
        evaluate()
    
