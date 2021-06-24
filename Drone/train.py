import numpy as np
import time
import tensorflow as tf
import os
import subprocess
import pdb
import json

# TFAgents Modules
import tf_agents
from tf_agents.drivers import dynamic_step_driver
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.environments import parallel_py_environment
from tf_agents.policies import random_tf_policy
from tf_agents.policies import policy_saver
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

# My modules
from Auxilaries.Extras import AdditionalMethods as extras
from Auxilaries.Extras.CustomTensorBoard import ModifiedTensorBoard
from Auxilaries.Environments.Environment import Environment
from Auxilaries.Agents import get_DQNorPPO as Agents
import Auxilaries.Environments.Additionals.RewardFunctions as Rewards
import Auxilaries.Extras.Parameters as Parameters


# Wrapper Definition
def train(PARAMETERS, EXPERIMENT_NAME, CONTINUE = False, ALREADY_PERFORMED_EPOCHS=0):

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    if PARAMETERS["Agent Type"] == 'hard' or PARAMETERS["Agent Type"] == 'random':
        print("You can't use this agent to train")

    with tf.device('/CPU:0'):
        """ ------- Parameters ------ """
        MULTI_DATA_COLLECTION           = False
        NUM_PARALLEL_ENVIRONMENTS       = 1
        EVAL_INTERVAL                   = 50
        MODELS_SAVED_TOTAL              = 5
        SAVE_DIRECTORY                  = "Auxilaries/TrainedModels/" + EXPERIMENT_NAME 

        # DQN
        DATA_COLLECTION_STEPS           = 500
        DATA_COLLECTION_STEPS_PER_EPOCH = 50
        REPLAY_BUFFER_MAX_LENGTH        = 5_000 
        BATCH_SIZE                      = 64

        # PPO
        if PARAMETERS['Agent Type'] == "ppo":
            REPLAY_BUFFER_MAX_LENGTH    = 50

        
        """-------------------- Create Environment --------------------"""

        
        # Create environment(s) and wrap them in TF Environments class
        if MULTI_DATA_COLLECTION == True:
            envs = [Environment] * NUM_PARALLEL_ENVIRONMENTS
            environment = tf_py_environment.TFPyEnvironment(parallel_py_environment.ParallelPyEnvironment(envs))
        else:
            env = Environment(  reward_fn       =PARAMETERS["Reward Function"],  
                                reset_method    =PARAMETERS["Reset Method"],
                                depth_on        =PARAMETERS["Depth Input"],
                                stack_on        =PARAMETERS["Stacked Input"],
                                environment     =PARAMETERS["Environment"],
                                name            =EXPERIMENT_NAME
                                )
            environment = tf_py_environment.TFPyEnvironment(env)
        

        """-------------------- Agent Creation --------------------"""
        global_step = tf.compat.v1.train.get_or_create_global_step()

        agent = Agents.getAgent(environment, PARAMETERS["Learning Rate"], global_step, PARAMETERS["Agent Type"])

        agent.initialize() 

        
        
        """-------------------- Replay Buffer --------------------"""
        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec       = agent.collect_data_spec,
            batch_size      = NUM_PARALLEL_ENVIRONMENTS,
            max_length      = REPLAY_BUFFER_MAX_LENGTH)


        """-------------------- Checkpoints --------------------"""
        checkpointer, policy_save   = extras.getCheckpointer(agent, replay_buffer, global_step, SAVE_DIRECTORY)
        policy_dir                  = SAVE_DIRECTORY + "/policies"


        """-------------------- Metrics --------------------"""
        
        metrics = extras.getMetrics(MULTI_DATA_COLLECTION, PARAMETERS['Agent Type'], NUM_PARALLEL_ENVIRONMENTS)
        observer = [replay_buffer.add_batch]

        # Setup Tensorboard
        name = "event" + EXPERIMENT_NAME
        board = ModifiedTensorBoard(name, log_dir=f"Auxilaries/TensorBoard/logs/{EXPERIMENT_NAME}")
        extras.printStatusStatement("Everything is ready! Start the Training Cycle")

        if not CONTINUE:
            with open(SAVE_DIRECTORY + '\parameters.txt','w') as f:
                f.writelines('{}:   {} \n'.format(k,v) for k, v in PARAMETERS.items())
                f.write('\n')

            if PARAMETERS["Agent Type"] == "dqn":
                stringlist = []
                agent._q_network.layers[0].summary(print_fn=lambda x: stringlist.append(x))
                short_model_summary = "\n".join(stringlist)
                print("Training agent with network: ", agent._q_network.layers[0].summary())
                with open(SAVE_DIRECTORY + '\\network_architectures.txt','w') as f:
                        f.write(short_model_summary)
        else:
            checkpointer.initialize_or_restore()
            global_step = tf.compat.v1.train.get_global_step()

        
        
        """-------------------- Driver --------------------"""    

        if PARAMETERS["Agent Type"] == "dqn":
            # Random Policy for data collection
            random_policy = random_tf_policy.RandomTFPolicy(environment.time_step_spec(),
                                                        environment.action_spec())

            # Setup RL Driver for Initial Data Collection
            collect_driver = dynamic_step_driver.DynamicStepDriver(environment, 
                                                            random_policy, 
                                                            observer, 
                                                            num_steps=DATA_COLLECTION_STEPS)

            # Fill our Replay Buffer with random experiences
            start = time.time()
            collect_driver.run()
            print("Time was ", time.time() - start, " seconds ")
        
            #Setup RL Driver for Training Data Collection
            train_driver = dynamic_step_driver.DynamicStepDriver(environment, 
                                                            agent.collect_policy, 
                                                            observer + metrics[0], 
                                                            num_steps=DATA_COLLECTION_STEPS_PER_EPOCH)

            eval_driver = dynamic_episode_driver.DynamicEpisodeDriver(environment, 
                                                            agent.policy, 
                                                            metrics[1], 
                                                            num_episodes= 10)

            # Transform Replay Buffer to Dataset
            dataset = replay_buffer.as_dataset(
                num_parallel_calls=3, 
                sample_batch_size=BATCH_SIZE, 
                num_steps=2).prefetch(3)
            iterator = iter(dataset)

        elif PARAMETERS['Agent Type'] == "ppo":

            # Create a driver for training and a driver for evaluation
            train_driver = dynamic_step_driver.DynamicStepDriver(environment, 
                                                            agent.collect_policy, 
                                                            observer + metrics[0], 
                                                            num_steps=int(REPLAY_BUFFER_MAX_LENGTH/NUM_PARALLEL_ENVIRONMENTS))
            
            eval_driver = dynamic_episode_driver.DynamicEpisodeDriver(environment, 
                                                            agent.policy, 
                                                            metrics[1], 
                                                            num_episodes= 10)
            #`as_dataset(..., single_deterministic_pass=True)` instead.

                                                                                

        """-------------------- Training --------------------"""
        # Optimize by wrapping some of the code in a graph using TF function.
        agent.train = common.function(agent.train)
        agent.train_step_counter.assign(0)

        
        # Training Loop
        for epoch in range(ALREADY_PERFORMED_EPOCHS, PARAMETERS["Epochs per Experiment"]):
            print("Epoch: ", epoch, end=' ')

            # Collect a few steps using collect_policy and save to the replay buffer.
            if not MULTI_DATA_COLLECTION:
                env.unPause()
            
            train_driver.run()
            
            if not MULTI_DATA_COLLECTION:
                env.setPause()

            # Sample a batch of data from the buffer and update the agent's network.
            if PARAMETERS['Agent Type'] == 'dqn':
                experience, unused_info = next(iterator)
            elif PARAMETERS['Agent Type'] == 'ppo':
                experience = replay_buffer.gather_all()
                replay_buffer.clear()

            # Train on GPU
            with tf.device('/GPU:0'):
                print("Train" , end=' ')
                train_loss = agent.train(experience).loss
                step = agent.train_step_counter.numpy()

            # Update training metrics
            extras.updateTensorboardTrain(PARAMETERS["Agent Type"], board, metrics, train_loss, epoch)
            
            # Perform evaluation step
            if epoch % EVAL_INTERVAL == 0 and epoch != 0:
                if not MULTI_DATA_COLLECTION:
                    env.unPause()
                eval_driver.run()
                extras.updateTensorboardEval(board, metrics, epoch)
                print("Evaluated", end=' ')

            # Save checkpoint
            if epoch % int(PARAMETERS["Epochs per Experiment"]/MODELS_SAVED_TOTAL) == 0 and epoch != 0:
                policy_save.save(policy_dir)
                print("SavedPolicy", end=' ')

            # Save this model in case of emergency
            checkpointer.save(global_step)

            # Keep track of how far we are
            lines = open(SAVE_DIRECTORY + '\parameters.txt','r').readlines()
            lines[-1] = str(epoch+1)
            open(SAVE_DIRECTORY + '\parameters.txt','w').writelines(lines)
            print("")

            
        """-------------------- Save -------------------- """
        # Save Policy
        print("Finished! Now saving....", end= ' ')
        policy_save.save(policy_dir)
        print("All done!")

        # End the simulation
        env.Kill("Blocks")

        # Mark this experiment as done
        lines = open(SAVE_DIRECTORY + '\parameters.txt','r').readlines()
        lines[-1] = "Done!"
        open(SAVE_DIRECTORY + '\parameters.txt','w').writelines(lines)


if __name__ == '__main__':
    extras.setGPUtoFraction(0.75)
    tf_agents.system.multiprocessing.enable_interactive_mode()


    default_parameters = {  "Learning Rate"             : 1e-3,
                            "Reward Function"           : Rewards.SimpleSparseClose,
                            "Reset Method"              : Parameters.Reset_Method.Directly_Behind_Person,
                            "Epochs per Experiment"     : 100,
                            "Agent Type"                : Parameters.Agent_Type.PPO_Agent,
                            "Stacked Input"             : True,
                            "Depth Input"               : True,
                            "Environment"               : 'BlocksObstacles'
                        }

    train(default_parameters, "Default")

  