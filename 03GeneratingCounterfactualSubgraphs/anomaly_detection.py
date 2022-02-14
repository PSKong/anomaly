from __future__ import division
from __future__ import print_function
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ""

import tensorflow as tf
from constructor import get_placeholder, get_model, get_optimizer, update
import numpy as np
from input_data import format_data
from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd
from preprocessing import construct_feed_dict
import scipy.io

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS


class AnomalyDetectionRunner():
    def __init__(self, settings):
        self.data_name = settings['data_name']
        self.iteration = settings['iterations']
        self.model = settings['model']


    def erun(self):
        model_str = self.model
        # load data
        feas = format_data(self.data_name)
        print("feature number: {}".format(feas['num_features']))

        # Define placeholders
        placeholders = get_placeholder()

        # construct model
        gcn_model = get_model(model_str, placeholders, feas['num_features'], feas['num_nodes'], feas['features_nonzero'])

        # Optimizer
        opt = get_optimizer(model_str, gcn_model, placeholders, feas['num_nodes'], FLAGS.alpha)

        # Initialize session
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # # Train model
        # for epoch in range(1, self.iteration+1):
        #
        #     reconstruction_errors, reconstruction_loss, attribute_reconstructions, structure_reconstructions = update(gcn_model, opt, sess, feas['adj_norm'], feas['adj_label'], feas['features'], placeholders, feas['adj'])
        #     if epoch % 10 == 0:
        #         print("Epoch:", '%04d' % (epoch), "train_loss=", "{:.5f}".format(reconstruction_loss))
        #
        #     if epoch % 100 == 0:
        #         y_true = [label[0] for label in feas['labels']]
        #         auc = roc_auc_score(y_true, reconstruction_errors)
        #         print(auc)
        #
        # sorted_errors = np.argsort(-reconstruction_errors, axis=0)
        #
        # with open('output/{}-ranking.txt'.format(self.data_name), 'w') as f:
        #     for index in sorted_errors:
        #         f.write("%s\n" % feas['labels'][index][0])
        #
        # df = pd.DataFrame({'AD-GCA':reconstruction_errors})
        # df.to_csv('output/{}-scores.csv'.format(self.data_name), index=False, sep=',')
        #
        # saveModel = tf.train.Saver(max_to_keep=2000)
        # saveModel.save(sess, 'data/test/savedModel/model_subGraph_24')


        saver = tf.train.import_meta_graph('./data/SavedModel/model_subGraph.meta') #'./data/test/savedModel/model_3719_5_1.meta'
        graph = tf.get_default_graph()

        saver.restore(sess, './data/SavedModel/model_subGraph') #'./data/test/savedModel/model_3719_5_1'
        reconstruction_errors_a, reconstruction_loss, attribute_reconstructions, structure_reconstructions = update(gcn_model, opt, sess, feas['adj_norm'],
                                                                 feas['adj_label'], feas['features'], placeholders,
                                                                 feas['adj'])

        #
        # emb_ = np.zeros((Sub_idx.shape[0], emb.shape[1])).astype(float)
        # for i in range(Sub_idx.shape[0]):
        #     emb_[i, :] = emb[Sub_idx[i], :]
        #
        # emb_1 = emb
        # idx = Sub_idx.tolist()
        # idx.sort(reverse=True)
        # for i in range(len(idx)):
        #     emb_1 = np.delete(emb_1, idx[i], 0)
        #
        # emb[0:Sub_idx.shape[0], :] = emb_
        # emb[16:emb.shape[0], :] = emb_1
        #
        # scipy.io.savemat('data/test/35_2/X_RealSubgraph_122.mat', {'X_RealSubgraph_122': emb})
        # scipy.io.savemat('data/test/35_1/RealSubgraph_index_35.mat', {'RealSubgraph_index_35': Sub_idx})

        scipy.io.savemat('data/GeneratedCounterfactualSubgraphs/CounterfactualSubgraphs_attribute.mat', {'CounterfactualSubgraphs_attribute': attribute_reconstructions})
        scipy.io.savemat('data/GeneratedCounterfactualSubgraphs/CounterfactualSubgraphs_structure.mat', {'CounterfactualSubgraphs_structure': structure_reconstructions})







