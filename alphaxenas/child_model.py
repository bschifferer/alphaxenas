import os
import sys
import numpy as np
import tensorflow as tf
import _pickle as pickle

import cv2
import time
from timeit import default_timer as timer

from alphaxenas.model_utils import get_weights, create_weights, log_to_textfile
from alphaxenas.data_utils import createPath, random_crop_and_flip, saveObjWithPickle

class ChildModel():
    """
    Child Model - convolutional neural network model, which will be retrained multiple times
    
    Functions:
        _apply_convcell (convcell, prev_cells, idx_nomberofconv, idx_convcell, list_trainable_weights, childname, bl_training, GLOBAL_WEIGHTS): Applies the convcell to the neural network
        build_model (GLOBAL_WEIGHTS): Builds the child network
        predict_batch (sess, tensors, input_data, y_output, bl_training, tmp_images, tmp_labels, batch_size, phase): Predict the data in batches
        predict_validation (images, labels, phase, initialize_new): Predict the accuracy of the input in batches
        couch_train (images, labels, max_noimprovements, max_iteration, lr_iteration_step, max_epochs, no_global_variables, safe_model): Trains the ChildModel
        
    Attributes:
        See incomment of __init__ function
    """
    def __init__(self, sess, images, labels, path, batch_size, convcell, global_ops, global_param, childname, variables_not_initialize, no_channels_start):
        """
        Function __init__

        Initialize the ChildModel model - save paramaters as attributes

        Args:
            sess (Tensorflow session): Tensorflow session, where the parameters are saved
            images (dict): Images used for training ChildModel
            labels (dict): Labels used for training ChildModel 
            path (str): Directory to store ChildModel files
            batch_size (int): Batch size for training Child Model 
            convcell (list): A list of operation for the convolutional cell
            global_ops (list): Not used anymore 
            global_param (list): List with some global parameters
            childname (str): The ChildModel name
            variables_not_initialize (list): List of tensorflow weights not to initialize
            no_channels_start (int): Number of default filter size
        
        Attributes:
            width (int): Number of image width
            height (int): Number of height height
            channels (int): Number of channels in image
            num_classes (int): Number of classes
        """
        self.sess = sess
        self.width = images['train'].shape[1]
        self.height = images['train'].shape[2]
        self.channels = images['train'].shape[3]
        self.num_classes = len(np.unique(labels['train']))
        self.path = path
        self.batch_size = batch_size
        self.global_ops = global_ops
        self.convcell = convcell
        self.global_param = global_param
        self.childname = childname
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)
        self.variables_not_initialize = variables_not_initialize
        self.no_channels_start = no_channels_start
    
    def _apply_convcell(self, convcell, prev_cells, idx_nomberofconv, idx_convcell, list_trainable_weights, childname, bl_training, GLOBAL_WEIGHTS):
        """
        Function _apply_convcell

        Applies the convcell to the neural network

        Args:
            convcell (list): A list of operation for the convolutional cell
            prev_cells (list): A list of output tensor of previous convolutional cells, which can be used as input
            idx_nomberofconv (int): Index of current convolutional cell batch (Not used anymore) 
            idx_convcell (int): Index of current convolutional cell
            list_trainable_weights (list): List of trainable weights
            childname (str): The ChildModel name
            bl_training (tf placeholder): Defines training/testing phase  
            GLOBAL_WEIGHTS (dict): Dictonary with shared tensorflow weights
        
        Attributes:
            final_out (tensorflow tensor): Output tensor
            list_trainable_weights (list): List of trainable weights
            bl_x1_used (boolean): If previous cell t-1 was used as input, then true 
            bl_x2_used (boolean): If previous cell t-2 was used as input, then true 
        """
        not_used_prev_cells = [x+2 for x in range(len(convcell))]
        bl_x1_used = False
        bl_x2_used = False
        print('-'*10)
        print(idx_nomberofconv, idx_convcell)
        idx = 0
        for convconfig in convcell:
            with tf.variable_scope('incell_' + str(idx)):
                idx += 1
                log_to_textfile(self.path + 'log.txt', str(convconfig) + '\n')
                log_to_textfile(self.path + 'log.txt', str(not_used_prev_cells) + '\n')
                if convconfig[0] == 0:
                    bl_x1_used = True
                if convconfig[1] == 1:
                    bl_x2_used = True
                with tf.variable_scope('incell_block1'):
                    if 'conv_' in convconfig[2]:
                        if prev_cells[convconfig[0]].get_shape()[3].value == self.no_channels_start:
                            h_1 = tf.nn.conv2d(prev_cells[convconfig[0]], GLOBAL_WEIGHTS[0][1][0][convconfig[2]][0], [1, 1, 1, 1], "SAME", data_format="NHWC")
                        if prev_cells[convconfig[0]].get_shape()[3].value == 2*self.no_channels_start:
                            h_1 = tf.nn.conv2d(prev_cells[convconfig[0]], GLOBAL_WEIGHTS[0][1][0][convconfig[2]][1], [1, 1, 1, 1], "SAME", data_format="NHWC")
                        h_1 = tf.layers.batch_normalization(h_1, training = bl_training)
                        h_1 = tf.nn.relu(h_1)
                    if 'convsep_' in convconfig[2]:
                        if prev_cells[convconfig[0]].get_shape()[3].value == self.no_channels_start:
                            h_1 = tf.nn.separable_conv2d(prev_cells[convconfig[0]], GLOBAL_WEIGHTS[0][1][0][convconfig[2]][0], GLOBAL_WEIGHTS[0][1][0][convconfig[2]][1], [1, 1, 1, 1], "SAME", data_format="NHWC")
                        if prev_cells[convconfig[0]].get_shape()[3].value == 2*self.no_channels_start:
                            h_1 = tf.nn.separable_conv2d(prev_cells[convconfig[0]], GLOBAL_WEIGHTS[0][1][0][convconfig[2]][2], GLOBAL_WEIGHTS[0][1][0][convconfig[2]][3], [1, 1, 1, 1], "SAME", data_format="NHWC")
                        h_1 = tf.layers.batch_normalization(h_1, training = bl_training)
                        h_1 = tf.nn.relu(h_1)
                    if 'id_' in convconfig[2]:
                        h_1 = prev_cells[convconfig[0]] 
                    if 'maxpool_3x3' == convconfig[2]:
                        h_1 = tf.layers.max_pooling2d(prev_cells[convconfig[0]], pool_size=(3,3), strides=(1,1), padding='same')
                    if 'maxpool_5x5' == convconfig[2]:
                        h_1 = tf.layers.max_pooling2d(prev_cells[convconfig[0]], pool_size=(5,5), strides=(1,1), padding='same')
                    if 'maxpool_7x7' == convconfig[2]:
                        h_1 = tf.layers.max_pooling2d(prev_cells[convconfig[0]], pool_size=(7,7), strides=(1,1), padding='same')
                    if 'avgpool_3x3' == convconfig[2]:
                        h_1 = tf.layers.average_pooling2d(prev_cells[convconfig[0]], pool_size=(3,3), strides=(1,1), padding='same')
                    if 'avgpool_5x5' == convconfig[2]:
                        h_1 = tf.layers.average_pooling2d(prev_cells[convconfig[0]], pool_size=(5,5), strides=(1,1), padding='same')
                    if 'avgpool_7x7' == convconfig[2]:
                        h_1 = tf.layers.average_pooling2d(prev_cells[convconfig[0]], pool_size=(7,7), strides=(1,1), padding='same')
                with tf.variable_scope('incell_block2'):
                    if 'convsep_' in convconfig[3]:
                        if prev_cells[convconfig[1]].get_shape()[3].value == self.no_channels_start:
                            h_2 = tf.nn.separable_conv2d(prev_cells[convconfig[1]], GLOBAL_WEIGHTS[0][1][0][convconfig[3]][0], GLOBAL_WEIGHTS[0][1][0][convconfig[3]][1], [1, 1, 1, 1], "SAME", data_format="NHWC")
                        if prev_cells[convconfig[1]].get_shape()[3].value == 2*self.no_channels_start:
                            h_2 = tf.nn.separable_conv2d(prev_cells[convconfig[1]], GLOBAL_WEIGHTS[0][1][0][convconfig[3]][2], GLOBAL_WEIGHTS[0][1][0][convconfig[3]][3], [1, 1, 1, 1], "SAME", data_format="NHWC")
                        h_2 = tf.layers.batch_normalization(h_2, training = bl_training, name = str(idx_convcell) + '_' + str(idx) +  '_batch2' + childname)
                        h_2 = tf.nn.relu(h_2)
                    if 'conv_' in convconfig[3]:
                        if prev_cells[convconfig[1]].get_shape()[3].value == self.no_channels_start:
                            h_2 = tf.nn.conv2d(prev_cells[convconfig[1]], GLOBAL_WEIGHTS[0][1][0][convconfig[3]][0], [1, 1, 1, 1], "SAME", data_format="NHWC")
                        if prev_cells[convconfig[1]].get_shape()[3].value == 2*self.no_channels_start:
                            h_2 = tf.nn.conv2d(prev_cells[convconfig[1]], GLOBAL_WEIGHTS[0][1][0][convconfig[3]][1], [1, 1, 1, 1], "SAME", data_format="NHWC")
                        h_2 = tf.layers.batch_normalization(h_2, training = bl_training)
                        h_2 = tf.nn.relu(h_2)
                    if 'id_' in convconfig[3]:
                        h_2 = prev_cells[convconfig[1]]
                    if 'maxpool_3x3' == convconfig[3]:
                        h_2 = tf.layers.max_pooling2d(prev_cells[convconfig[1]], pool_size=(3,3), strides=(1,1), padding='same')
                    if 'maxpool_5x5' == convconfig[3]:
                        h_2 = tf.layers.max_pooling2d(prev_cells[convconfig[1]], pool_size=(5,5), strides=(1,1), padding='same')
                    if 'maxpool_7x7' == convconfig[3]:
                        h_2 = tf.layers.max_pooling2d(prev_cells[convconfig[1]], pool_size=(7,7), strides=(1,1), padding='same')
                    if 'avgpool_3x3' == convconfig[3]:
                        h_2 = tf.layers.average_pooling2d(prev_cells[convconfig[1]], pool_size=(3,3), strides=(1,1), padding='same')
                    if 'avgpool_5x5' == convconfig[3]:
                        h_2 = tf.layers.average_pooling2d(prev_cells[convconfig[1]], pool_size=(5,5), strides=(1,1), padding='same')
                    if 'avgpool_7x7' == convconfig[3]:
                        h_2 = tf.layers.average_pooling2d(prev_cells[convconfig[1]], pool_size=(7,7), strides=(1,1), padding='same')
                if convconfig[4] == 'add':
                    if h_1.get_shape()[3].value != self.no_channels_start:
                            w = get_weights(childname + '_' + str('tmpid1_') + str(idx_nomberofconv) + '_' + str(idx_convcell), [1, 1, h_1.get_shape()[3].value, self.no_channels_start])
                            list_trainable_weights.append(w)
                            h_1 = tf.nn.conv2d(h_1, w, [1, 1, 1, 1], "SAME")
                            h_1 = tf.layers.batch_normalization(h_1, training = bl_training)
                            h_1 = tf.nn.relu(h_1)
                    if h_2.get_shape()[3].value != self.no_channels_start:
                            w = get_weights(childname + '_' + str('tmpid2_') + str(idx_nomberofconv) + '_' + str(idx_convcell), [1, 1, h_2.get_shape()[3].value, self.no_channels_start])
                            list_trainable_weights.append(w)
                            h_2 = tf.nn.conv2d(h_2, w, [1, 1, 1, 1], "SAME")
                            h_2 = tf.layers.batch_normalization(h_2, training = bl_training)
                            h_2 = tf.nn.relu(h_2)
                    h_out = tf.add(h_1, h_2)
                elif convconfig[4] == 'concat':
                    h_out = tf.concat([h_1, h_2], axis=-1)
                else:
                    print('Error')
                prev_cells.append(h_out)
                if not('id_' in convconfig[2]) and not('pool' in convconfig[2]):
                    list_trainable_weights.append(GLOBAL_WEIGHTS[0][1][0][convconfig[2]])
                if not('id_' in convconfig[3]) and not('pool' in convconfig[3]):
                    list_trainable_weights.append(GLOBAL_WEIGHTS[0][1][0][convconfig[3]])
                if convconfig[0] in not_used_prev_cells:
                    not_used_prev_cells.remove(convconfig[0])
                if convconfig[1] in not_used_prev_cells:
                    not_used_prev_cells.remove(convconfig[1])
        
        with tf.variable_scope('celloutput_' + str(idx_convcell)):
            final_out = tf.concat([prev_cells[i] for i in not_used_prev_cells], axis=-1)
            shp_in = final_out.get_shape()[3].value
            w = get_weights(childname + '_' + str('final_') + str(idx_nomberofconv) + '_' + str(idx_convcell), [1, 1, shp_in, self.global_param[4]])
            list_trainable_weights.append(w)
            final_out = tf.nn.conv2d(final_out, w, [1, 1, 1, 1], "SAME")
            final_out = tf.layers.batch_normalization(final_out, training = bl_training)
            final_out = tf.nn.relu(final_out)    
        return(final_out, list_trainable_weights, bl_x1_used, bl_x2_used)

            
    def build_model(self, GLOBAL_WEIGHTS):
        """
        Function build_model

        Build the ChildModel model

        Args:
            GLOBAL_WEIGHTS (dict): Dictonary with shared tensorflow weights
        """
        input_data = tf.placeholder(shape=[None, 32, 32, 3], dtype=tf.float32, name='x_input')
        y_output = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='y_output')
        bl_training = tf.placeholder(tf.bool, name='training')
        learning_rate = tf.placeholder(tf.float32, shape=[])
        
        N_numberofconv = self.global_param[0]
        N_convcells = self.global_param[1]
        B = self.global_param[2]
        action_space = self.global_param[3]
        no_channels_start = self.global_param[4]
        
        childname = self.childname
        convcell = self.convcell
        
        list_trainable_weights = []
        list_concat = []
        
        x_1 = tf.nn.conv2d(input_data, GLOBAL_WEIGHTS[-1][0], [1, 1, 1, 1], "SAME", name = childname + '_first_conv')
        x_1 = tf.layers.batch_normalization(x_1, training = bl_training, name = childname + '_first_batch')
        x_1 = tf.nn.relu(x_1, name = childname + '_first_relu')    
        x_2 = x_1
        list_trainable_weights.append(GLOBAL_WEIGHTS[-1][0])
        
        for i in range(N_convcells):
            with tf.variable_scope('cell_' + str(i)):
                final_out, list_trainable_weights, x_1_used, x_2_used = self._apply_convcell(convcell, [x_1, x_2], 1, i, list_trainable_weights, childname, bl_training, GLOBAL_WEIGHTS)
                log_to_textfile(self.path + 'log.txt', str(x_1_used) + '\n')
                log_to_textfile(self.path + 'log.txt', str(x_2_used) + '\n')
                if not(x_1_used):
                    list_concat.append(x_1)
                if not(x_2_used):
                    list_concat.append(x_2)
                x_2 = x_1
                x_1 = final_out
        
        list_concat.append(final_out)
        log_to_textfile(self.path + 'log.txt', str(list_concat) + '\n')
        if len(list_concat)>1:
            x = tf.concat(list_concat, axis=-1)
        else:
            x = final_out
        x = tf.reduce_mean(x, axis=[1,2])
        log_to_textfile(self.path + 'log.txt', str(x) + '\n')
        x = tf.layers.dense(x, 10, kernel_regularizer=self.regularizer)
        x_softmax = tf.nn.softmax(x)
        
        y = tf.one_hot(tf.cast(y_output, tf.int32), self.num_classes)
        l2_loss = tf.losses.get_regularization_loss()
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=x, labels=y))
        loss += l2_loss
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        list_trainable_weights +=  [x for x in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if 'dense' in str(x.name)] 
        
        self.loss = loss
        self.input_data = input_data
        self.y_output = y_output
        self.x = x
        self.bl_training = bl_training
        self.x_softmax = x_softmax
        self.learning_rate = learning_rate
        self.list_trainable_weights = list_trainable_weights
    
    def predict_batch(self, sess, tensors, input_data, y_output, bl_training, tmp_images, tmp_labels, batch_size, phase = 0):
        """
        Function predict_batch

        Predict the data in batches as it could be too big to fit in the GPU memory, at once.

        Args:
            sess (Tensorflow session): Tensorflow session, where the parameters are saved
            tensors (list): List of tensorflow tensor, which should be predicted
            input_data (): Not used
            y_output (): Not used
            bl_training (tf placeholder): Defines training/testing phase
            tmp_images (np array): Images to predict
            tmp_labels (np array): Labels for prediction
            batch_size (int): Batchsize
            phase (int): Input for bl_training (training/testing phase)
        
        Attributes:
            y_tmp_final (list): List of true labels
            x_tmp_pred_final (list): List of predicted labels
            tmp_o_loss_final (list): List of loss terms
        """
        sess = self.sess
        idx_tmp = np.asarray(range(tmp_images.shape[0]))
        no_batches_tmp = idx_tmp.shape[0]//batch_size + 1
        y_tmp_final = []
        x_tmp_pred_final = []
        tmp_o_loss_final = []
        for batch in range(no_batches_tmp-1):
            x_tmp_batch = tmp_images[idx_tmp[(batch*batch_size):(min((1+batch)*batch_size,tmp_images.shape[0]))]]
            y_tmp_batch = tmp_labels[idx_tmp[(batch*batch_size):(min((1+batch)*batch_size,tmp_images.shape[0]))]]
            tmp_o_loss, tmp_o_x = sess.run(tensors, feed_dict={input_data:x_tmp_batch, y_output: y_tmp_batch, bl_training: phase})
            arg_max = np.argmax(tmp_o_x, axis=-1)
            x_tmp_pred_final.append(arg_max)
            y_tmp_final.append(y_tmp_batch)
            tmp_o_loss_final.append(tmp_o_loss)
            
        return(y_tmp_final, x_tmp_pred_final, tmp_o_loss_final)

    def predict_validation(self, images, labels, phase = 0, initialize_new = True):
        """
        Function predict_validation

        Predict the data in batches as it could be too big to fit in the GPU memory, at once.

        Args:
            images (np array): Images to predict
            labels (np array): Labels for prediction
            phase (int): Input for bl_training (training/testing phase)
            initialize_new (boolean): Initialize not shared variables new
        
        Attributes:
            acc (float): Accuracy of prediction
        """
        sess = self.sess
        loss = self.loss
        input_data = self.input_data 
        y_output = self.y_output
        x = self.x
        bl_training = self.bl_training
        x_softmax = self.x_softmax
        batch_size = self.batch_size
        path = self.path
        
        x_images = images
        y_label = labels
        y_label = y_label.reshape((-1, 1))
        
        iteration = 0
        iteration_total = []
        val_acc_total = []
        test_acc_total = []
        total_time_total = []
        result = {}
        
        if initialize_new:
            variables_to_initialize = [x for x in tf.global_variables() if not(x.name in [y.name for y in self.variables_not_initialize])]
            sess.run(tf.variables_initializer(variables_to_initialize))
        
        y_final, x_pred_final, loss_final = self.predict_batch(sess, 
                                                               [loss, x_softmax],
                                                               input_data, 
                                                               y_output, 
                                                               bl_training,
                                                               x_images, 
                                                               y_label, 
                                                               batch_size,
                                                               phase)
        
        y_final = np.asarray(y_final)
        x_pred_final = np.asarray(x_pred_final)
        acc = np.mean(x_pred_final.reshape((-1,1))==y_final.reshape((-1,1)))
        loss_final = np.mean(loss_final)        
        log_to_textfile(self.path + 'log.txt', 'Validation loss: ' + str(loss_final) + 'Validation acc: ' + str(acc) + '\n')
        return(acc)

    def reload_model(self, sess, path):
        """
        Function reload_model

        Loads weights of a child model

        Args:
            sess (Tensorflow session): Tensorflow session, where the parameters are saved
            path (str): Path of the child model weights
        """
        saver = tf.train.Saver()
        saver.restore(sess, path)
    
    def couch_train(self, images, labels, max_noimprovements, max_iteration, lr_iteration_step, max_epochs, no_global_variables = False, safe_model = False):
        """
        Function couch_train

        Predict the data in batches as it could be too big to fit in the GPU memory, at once.

        Args:
            images (dict): Images used for training ChildModel
            labels (dict): Labels used for training ChildModel 
            max_noimprovements (int): Stop training ChildModel if does not improve over number of epochs
            max_iteration (int): Number of maximal training steps in ChildModel
            lr_iteration_step (list): Epochs when learning rate is decayed in ChildModel
            max_epochs (int): Number of epochs for training ChildModel
            no_global_variables (boolean): If True, no global variables are trained in ChildModel
            safe_model (boolean): If True, then ChildModel weights are safed
        """
        # Initialize / store Child attributes in local variables
        sess = self.sess
        loss = self.loss
        input_data = self.input_data 
        y_output = self.y_output
        x = self.x
        bl_training = self.bl_training
        x_softmax = self.x_softmax
        batch_size = self.batch_size
        lr = 0.1
        learning_rate = self.learning_rate
        path = self.path
        list_trainable_weights = self.list_trainable_weights
        
        createPath(path)
        createPath(path + '/model')
        x_train = images['train']
        y_train = labels['train']
        x_val = images['valid']
        y_val = labels['valid']
        y_train = y_train.reshape((-1, 1))
        y_val = y_val.reshape((-1, 1))
        
        # Creating some variables to store results
        iteration = 0
        iteration_total = []
        val_acc_total = []
        total_time_total = []
        result = {}
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        
        # Reinitialize child specific variables
        if no_global_variables:
            list_trainable_weights = [x for x in list_trainable_weights if not(x.name in [y.name for y in self.variables_not_initialize])]
        
        # Create optimnizer
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True).minimize(loss, var_list=list_trainable_weights)
        
        variables_to_initialize = [x for x in tf.global_variables() if not(x.name in [y.name for y in self.variables_not_initialize])]
        
        best_loss = 999
        best_acc = 0
        best_epoch = 0
        bl_break = False
        
        saver = tf.train.Saver()
        # Initialize only new variables (not shared)
        sess.run(tf.variables_initializer(variables_to_initialize))
        createPath(self.path + "/log/")
        createPath(self.path + "/model/")
        # Create writer for tensorboard
        writer = tf.summary.FileWriter(self.path + "/log/{}".format(self.childname), sess.graph)
        idx = np.asarray(range(x_train.shape[0]))
        counter_run_noimprovement = 0
        # Epoch iterations
        for e in range(max_epochs):
            if bl_break:
                break
            start = timer()
            no_batches = idx.shape[0]//batch_size + 1
            np.random.shuffle(idx)
            # One training loop
            for batch in range(no_batches-1):
                x_train_batch = x_train[idx[(batch*batch_size):(min((1+batch)*batch_size,x_train.shape[0]))]]
                x_train_batch = random_crop_and_flip(x_train_batch)
                y_train_batch = y_train[idx[(batch*batch_size):(min((1+batch)*batch_size,x_train.shape[0]))]]
                o_loss, o_optimizer, o_x = sess.run([loss, optimizer, x], feed_dict={input_data:x_train_batch, 
                                                                                     y_output:y_train_batch, 
                                                                                     bl_training: 1,
                                                                                     learning_rate: lr})
                if iteration == max_iteration:
                    bl_break = True
            
            iteration += 1
            # Learning rate decay
            if iteration in lr_iteration_step:
                lr = 0.1 * lr
                log_to_textfile(self.path + 'log.txt', 'New learning rate: ' + str(lr) + '\n')

            # Predict validation set
            y_val_final, x_val_pred_final, val_o_loss_final = self.predict_batch(sess, 
                                                                                 [loss, x_softmax],
                                                                                 input_data, y_output, bl_training,
                                                                                 x_val, 
                                                                                 y_val, 
                                                                                 batch_size) 

            y_val_final = np.asarray(y_val_final)
            x_val_pred_final = np.asarray(x_val_pred_final)
            val_acc = np.mean(x_val_pred_final.reshape((-1,1))==y_val_final.reshape((-1,1)))
            val_o_loss_mean = np.mean(val_o_loss_final)
            # Save best model
            if best_acc < val_acc:
                log_to_textfile(self.path + 'log.txt', 'Safe best model' + '\n')
                best_acc = val_acc
                best_epoch = e
                counter_run_noimprovement = 0
                if safe_model:
                    saver.save(sess, self.path + 'model/{}'.format(self.childname))
            else:
                counter_run_noimprovement = counter_run_noimprovement+1

            end = timer()
            total_time = end-start
            log_to_textfile(self.path + 'log.txt', 'Time: ' + str(total_time) + ' Epoch: ' + str(e) + ' Iteration: ' + str(iteration) + ' No Improv: ' + str(counter_run_noimprovement) + ' Val Loss: ' + str(val_o_loss_mean) + ' Best Acc: ' + str(best_acc) + ' Val Acc: ' + str(val_acc) + '\n')
            # Break training if the model does not improve over certain number of epochs
            if counter_run_noimprovement>max_noimprovements:
                bl_break = True