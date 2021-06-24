import tf_agents
import random
import tensorflow as tf
import sys

from Auxilaries.Environments.Environment import Environment
import Auxilaries.Environments.Additionals.RewardFunctions as Rewards
import Auxilaries.Extras.Parameters as Parameters
from Auxilaries.Extras import AdditionalMethods as extras
from train import train

from numpy.random import seed 


if __name__ == '__main__':
    
    # Setup GPU
    extras.setGPUtoFraction(0.75)
    tf_agents.system.multiprocessing.enable_interactive_mode()

    # Parameters and Experiments
    global_parameters = {   "Learning Rate"             : 1e-3,
                            "Reward Function"           : Rewards.SimpleSparseClose,
                            "Reset Method"              : Parameters.Reset_Method.Directly_Behind_Person,
                            "Epochs per Experiment"     : 2000}
    
    environment_list = Parameters.Environments.FullList
    if len(sys.argv) > 1:
        environment_list = [sys.argv[1]]
        
    for environment in environment_list:
        experiments = { 
                        environment + " Run 3 - DQN nostack-normal" :{"Agent Type" : Parameters.Agent_Type.DQN_Agent,
                                                            "Stacked Input"         : True,
                                                            "Depth Input"           : False,
                                                            "Environment"   	    : environment}
        }

        random.seed(1)
        seed(1)
        tf.compat.v1.set_random_seed(1)

        
        # Perform Experiments
        for experiment in experiments:
            # Check if this experiment has been run at all. If not: run it
            try:
                my_file = open("Auxilaries/TrainedModels/"+experiment+"/parameters.txt")
            except: 
                train({**experiments[experiment], **global_parameters}, experiment)

            my_file = open("Auxilaries/TrainedModels/"+experiment+"/parameters.txt")
            # Check if this experiment has been finished, if not: continue
            last_line = my_file.readlines()[-1]
            if last_line == "Done!":
                continue
            else:
                continue_from_epoch = int(last_line)
                train({**experiments[experiment], **global_parameters}, experiment, CONTINUE=True, ALREADY_PERFORMED_EPOCHS=continue_from_epoch)

