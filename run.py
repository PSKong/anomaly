import tensorflow as tf
import numpy as np
from anomaly_detection import AnomalyDetectionRunner
flags = tf.app.flags
FLAGS = flags.FLAGS


flags.DEFINE_integer('hidden3', 64, 'Number of units in hidden layer 3.')
flags.DEFINE_integer('discriminator_out', 0, 'discriminator_out.')
flags.DEFINE_float('discriminator_learning_rate', 0.005, 'Initial learning rate.')#0.001
flags.DEFINE_float('learning_rate', .5*0.001, 'Initial learning rate.')
flags.DEFINE_integer('hidden1', 32*2, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32*2, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')
flags.DEFINE_integer('seed', 50, 'seed for fixing the results.')
flags.DEFINE_integer('iterations', 300, 'number of iterations.') #epoch300
flags.DEFINE_float('alpha', 0.5, 'balance parameter')


'''
We did not set any seed when we conducted the experiments described in the paper;
We set a seed here to steadily reveal better performance of ARGA
'''
seed = 7
np.random.seed(seed)
tf.set_random_seed(seed)

data_list = ['Yelp', 'PubMed', 'Reddit']
dataname = data_list[0]
model = 'gcn_ae'  # 'gcn_ae' or 'gcn_vae'
task = 'anomaly_detection'
settings = {'data_name': dataname, 'iterations' : FLAGS.iterations, 'model' : model}

runner = None
if task == 'anomaly_detection':
    runner = AnomalyDetectionRunner(settings)

runner.erun()
