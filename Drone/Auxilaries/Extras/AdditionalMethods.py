from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tf_agents.utils import common
from tf_agents.policies import policy_saver
from tf_agents.metrics import tf_metrics
import numpy as np

""" My own print statement """
def printStatusStatement(text):
  print("\n \n---------------------------- ", text, " ----------------------------\n \n")

""" Limit Tensorflow allocation space on GPU """
def setGPUtoFraction(fraction = 0.666):
    config = ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.666
    session = InteractiveSession(config=config)

""" Create the checkpointers so the models can be saved """
def getCheckpointer(agent, replay_buffer, global_step, directory):

  # Checkpoint initialization
  checkpoint_dir = directory + '/checkpoints'
  checkpointer = common.Checkpointer(
      ckpt_dir        = checkpoint_dir,
      max_to_keep     = 1,
      agent           = agent,
      policy          = agent.policy,
      replay_buffer   = replay_buffer,
      global_step     = global_step
  )

  # Policy Initialization
  policy_save = policy_saver.PolicySaver(agent.policy)
  return checkpointer, policy_save

""" Create the required metrics for the type of learning that will happen """  
def getMetrics(mdc, agent, num_batch):
  assert agent == "dqn" or agent == "ppo"

  if mdc == True:
    train_average_return                 = tf_metrics.AverageReturnMetric(batch_size=num_batch)
    train_average_episode_length         = tf_metrics.AverageEpisodeLengthMetric(batch_size=num_batch) 
    eval_average_return                  = tf_metrics.AverageReturnMetric(batch_size=num_batch)
    eval_max_return                      = tf_metrics.MaxReturnMetric(batch_size=num_batch)
    eval_min_return                      = tf_metrics.MinReturnMetric(batch_size=num_batch)

    train_metrics   = [train_average_return, train_average_episode_length]
    eval_metrics    = [eval_average_return, eval_max_return, eval_min_return]
    metrics         = [train_metrics, eval_metrics]

    if agent == 'dqn':
      train_metrics.append(tf_metrics.MaxReturnMetric(batch_size=num_batch))
      train_metrics.append(tf_metrics.MinReturnMetric(batch_size=num_batch))

  else:
    train_average_return            = tf_metrics.AverageReturnMetric()
    train_average_episode_length    = tf_metrics.AverageEpisodeLengthMetric()
    eval_average_return             = tf_metrics.AverageReturnMetric()
    eval_max_return                 = tf_metrics.MaxReturnMetric()
    eval_min_return                 = tf_metrics.MinReturnMetric()

    train_metrics    = [train_average_return, train_average_episode_length]
    eval_metrics     = [eval_average_return, eval_max_return, eval_min_return]
    metrics          = [train_metrics, eval_metrics]

    if agent == 'dqn':
      train_metrics.append(tf_metrics.MaxReturnMetric())
      train_metrics.append(tf_metrics.MinReturnMetric())

  return metrics

""" Update Tensorboard during training time """
def updateTensorboardTrain(agent, board, metrics, loss, epoch):
  assert agent == "dqn" or agent == "ppo"

  for metric in metrics[0]:
    if metric.result() >= 2e6 or metric.result() <= -2e6:
      metric.reset()

  if agent == 'ppo':

    board.update_stats( TrainingLoss            = round(float(loss),2),
                        TrainAverageReturn      = round(float(metrics[0][0].result()), 2),
                        TrainAVGEpisodeLength   = int(metrics[0][1].result()), step = epoch)
    for metric in metrics[0]:
      metric.reset()

  if agent == "dqn":
    board.update_stats( TrainingLoss            = round(float(loss),2),
                        TrainAverageReturn      = round(float(metrics[0][0].result()),2),
                        TrainAVGEpisodeLength   = int(metrics[0][1].result()),
                        TrainMaxReturn          = int(metrics[0][2].result()),
                        TrainMinReturn          = int(metrics[0][3].result()),
                        step = epoch)

""" Update Tensorboard during test time """  
def updateTensorboardEval(board, metrics, epoch):

  board.update_stats(EvalAverageReturn          = int(metrics[1][0].result()), 
                                EvalMaxReturn   = int(metrics[1][1].result()), 
                                EvalMinReturn   = int(metrics[1][2].result()),
                                step = epoch)
  for metric in metrics[1]:
      metric.reset()

""" Creates a dictionary from a file """ 
def transformToDict(filename):
  f = open(filename, 'r')
  lines = f.readlines()[0:-1]
  dictionary = {}
  for line in lines:
    splits = line.split(":")
    if splits[0] != "Reward Function":
      new_value = splits[1].replace("\n", "").replace(" ", "")
    else:
      new_value = splits[1].replace("\n", "").split()[1] 
    dictionary[splits[0]] = new_value
  return dictionary
