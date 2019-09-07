import tensorflow as tf
import numpy as np

from alphaxenas.data_utils import read_data, saveObjWithPickle, createPath
from alphaxenas.model_utils import get_weights, create_weights, log_to_textfile
from alphaxenas.child_model import ChildModel
from alphaxenas.controller_lstm import LSTMModel
from alphaxenas.controller_mcts import MCTS

from alphaxenas.model_utils import legal_action_from_seq, one_hot, s_to_convcell
import random
import pickle

class Coach():
    """
    Coach class - manage exploration of MCTS, LSTM, sharing global weigths
    
    Functions:
        _reset_graph (): Resets the tensorflow session and reloads shared global weights
        train (): Trains the Coach, MCTS and ChildModels

    Attributes:
        See incomment of __init__ function
    """
    def __init__(self, 
                 path, 
                 GLOBAL_WEIGHTS, 
                 num_learning_iteration, 
                 num_expansions,
                 B,
                 action_space, 
                 combine_op,
                 size_lstm, 
                 sess,
                 max_replay_size, 
                 images, 
                 labels, 
                 global_ops, 
                 global_param, 
                 no_channels_start,
                 search_perc_range,
                 search_epoch_range,
                 N_convcells_range, 
                 no_for_uniform, 
                 new_mcts_every_i, 
                 alphax_version, 
                 debug_no_trainig):
        """
        Function __init__

        Initialize the Couach - save paramaters as attributes

        Args:
            path (str): Directory to store all files
            GLOBAL_WEIGHTS (dict): Dictonary with shared tensorflow weights
            num_learning_iteration (int): Number of iteration to train 
            num_expansions (int): Number of tree search for a given state
            B (int): Number of inconv cells
            action_space (dict): Dictionary with possible neural network operations + weights
            combine_op (dict): Dictionary with possible combine operations + weights
            size_lstm (int): Size of hidden layer in LSTM; 
            sess (None): Not used anymore
            max_replay_size (int): Number of replay size to train LSTM
            images (dict): Images used for training ChildModel
            labels (dict): Labels used for training ChildModel 
            global_ops (list): Not used anymore 
            global_param (list): List with some global parameters
            no_channels_start (int): Number of channels in convolution layer
            search_perc_range (list): Percentage range of data used for child model (only first value is used)
            search_epoch_range (list): Number of epochs range for child model (only first value is used)
            N_convcells_range (int): Number of convcells used
            no_for_uniform (int): Number of learning iteration (Tree), which uses a uniform distribution for Ps
            new_mcts_every_i (int): Every number of learning iteration (Tree), reinitizalize MCTS tree
            alphax_version (boolean): If True other MCTS formula is used (based on AlphaX paper)
            debug_no_trainig (boolean): If True no ChildModel are trained and only dummy value is returned
        
        Attributes:
            num_actions (int): Number of total actions possible (used for LSTM prediction)
            full_examples (list): Stores replay for training LSTM
            max_noimprovements (int): Stop training ChildModel if does not improve over number of epochs
            max_iteration (int): Number of maximal training steps in ChildModel
            lr_iteration_step (list): Epochs when learning rate is decayed in ChildModel
            max_epochs (int): Number of epochs for training ChildModel
            no_global_variables (boolean): If True, no global variables are trained in ChildModel
            lstm_batchsize (int): Batchsize for training LSTM model
            lstm_learning_rate (float): Learning rate for LSTM model
            no_trained_in_search (int): Counter for number of ChildModel in MCTS Search trained
            no_trained_final (int): Counter for number of ChildModel trained in the end of one iteration_learning
        """
        self.no_channels_start = no_channels_start
        self.path = path
        self.num_learning_iteration = num_learning_iteration
        self.num_expansions = num_expansions
        self.B = B
        self.action_space = action_space
        self.combine_op = combine_op
        self.size_lstm = size_lstm
        self.sess = sess
        
        self.num_actions = B + len(action_space.keys()) + len(combine_op.keys())
        
        self.full_examples = []
        self.max_replay_size = max_replay_size
        
        self.images = images
        self.labels = labels
        self.images_total = images
        self.labels_total = labels
        self.global_ops = global_ops
        self.global_param = global_param.copy()
        self.global_param_org = global_param.copy()
        self.current_nconv = N_convcells_range[0]
        
        self.max_noimprovements = 2
        self.max_iteration = 100000 
        self.lr_iteration_step =  [2, 4]
        self.max_epochs = 20 
        self.no_global_variables = False
        
        self.lstm_batchsize = 32
        self.lstm_learning_rate = 0.01
        self.no_trained_in_search = 0
        self.no_trained_final = 0
        
        self.search_perc_range = search_perc_range
        self.search_epoch_range = search_epoch_range
        self.N_convcells_range = N_convcells_range
        
        createPath(path)
        createPath(path + '/Coach/')
        createPath(path + '/Coach/global_weights')
        createPath(path + '/Children/')
        createPath(path + '/Children/Finaltrain/')
        createPath(path + '/Children/Insearchtrain/')
        createPath(path + '/LSTM/')
        
        self._reset_graph(True)
        self.sess.run(tf.global_variables_initializer())
        self.no_for_uniform = no_for_uniform
        self.new_mcts_every_i = new_mcts_every_i
        self.alphax_version = alphax_version
        self.debug_no_trainig = debug_no_trainig
        
    def _reset_graph(self, first_initialize = False):
        """
        Function _reset_graph

        Resets the tensorflow graph and restores the shared global weights

        Args:
            first_initialize (boolean): If True, shared global weights are not loaded
        """
        self.global_param = self.global_param_org.copy()
        tf.reset_default_graph()
        self.sess = tf.Session()
        self.GLOBAL_WEIGHTS = None
        with tf.variable_scope('GLOBAL_WEIGHTS'): 
            GLOBAL_WEIGHTS = create_weights(self.global_param[0], self.global_param[1], self.B, self.action_space, self.global_param[4])
            GLOBAL_WEIGHTS[-1] = {}
            GLOBAL_WEIGHTS[-1][0] = get_weights(str('Start_1'), [1, 
                                                                 1, 
                                                                 3, 
                                                                 self.no_channels_start])
            GLOBAL_WEIGHTS[-2]  = {}

        self.GLOBAL_WEIGHTS = GLOBAL_WEIGHTS
        self.lstm_model = LSTMModel(self.sess, self.num_actions, self.num_actions, self.size_lstm, self.B, self.path + '/LSTM/')
        self.variables_not_initialize = [x for x in tf.global_variables()]
        self.loader = tf.train.Saver(max_to_keep=10, var_list=self.variables_not_initialize)
        if not(first_initialize):
            self.loader.restore(self.sess, self.path + '/Coach/global_weights/global_weight')
        self.global_param[1] = self.current_nconv

    def train(self):
        """
        Function train

        Trains the Coach, MCTS and ChildModels

        Args:
            
        """
        lstm_perc = 0.05
        lstm_epoch = 2
        unique_idx = 0
        total_results = []
        no_mcts = 0
        # First initialization of MCTS tree
        mcts = MCTS(self.num_actions, 
                    self.B, 
                    len(self.action_space.keys()), 
                    len(self.combine_op.keys()), 
                    self.action_space, 
                    self.combine_op,
                    self.path + '/Coach/' + 'logs.txt',
                    self.no_channels_start)
        # Iteration of searches
        for i in range(self.num_learning_iteration):
            print(i)
            use_uniform = False
            if self.no_for_uniform > i:
                use_uniform = True
            #if i % (int(self.num_learning_iteration / len(range(self.N_convcells_range[1]-self.N_convcells_range[0])))) == 0:
            #self.current_nconv = min(self.current_nconv+1,self.N_convcells_range[1])
            self.global_param[1] = self.current_nconv
                
            #lstm_perc = random.uniform(self.search_perc_range[0], self.search_perc_range[1])
            #lstm_epoch = random.randint(self.search_epoch_range[0], self.search_epoch_range[1])
            lstm_perc = self.search_perc_range[0]
            lstm_epoch = self.search_epoch_range[0]
            tmp_result = []
            
            # Only use X percent for training ChildModels
            N = self.images_total['train'].shape[0]
            idx = np.asarray(range(N))
            np.random.shuffle(idx)
            images = self.images_total.copy()
            labels = self.labels_total.copy()
            images['train'] = self.images_total['train'][idx[0:int(lstm_perc*N)],:,:,:].copy()
            labels['train'] = self.labels_total['train'][idx[0:int(lstm_perc*N)]].copy()
            
            # Logging to file
            log_to_textfile(self.path + '/Coach/' + 'logs.txt', '###################################### New Search ###################################### \n')
            log_to_textfile(self.path + '/Coach/' + 'logs.txt', 'Trained final: ' + str(self.no_trained_final) + ' Trained in search:' + str(self.no_trained_in_search) + ' \n')
            log_to_textfile(self.path + '/Coach/' + 'logs.txt', 'LSTM percentage: ' + str(lstm_perc) + ' LSTM Epoch:' + str(lstm_epoch) + ' Size:' + str(images['train'].shape) + ' \n')
            log_to_textfile(self.path + '/Coach/' + 'logs.txt', 'NConvcell: ' + str(self.current_nconv) + ' \n')
            
            if (i>0) and (i % self.new_mcts_every_i == 0) and not(self.alphax_version):
                # Reinitialize MCTS tree after new_mcts_every_i iterations
                log_to_textfile(self.path + '/Coach/' + 'logs.txt', 'New MCTS \n')
                mcts = MCTS(self.num_actions, 
                            self.B, 
                            len(self.action_space.keys()), 
                            len(self.combine_op.keys()), 
                            self.action_space, 
                            self.combine_op,
                            self.path + '/Coach/' + 'logs.txt',
                            self.no_channels_start)
                no_mcts = no_mcts + 1
            
            # Logging to dictionary
            tmp123_result = {}
            tmp123_result['i'] = i
            tmp123_result['no_trained_final'] = self.no_trained_final
            tmp123_result['no_trained_search'] = self.no_trained_in_search
            tmp123_result['lstm_perc'] = lstm_perc
            tmp123_result['lstm_epoch'] = lstm_epoch
            tmp123_result['nconvcell'] = self.current_nconv
            tmp123_result['no_mcts'] = no_mcts
            
            # Initial state
            s = (0,0,)
            examples = []
            
            # Logging to file
            # Logging to dictionary
            tmp123_result['in_search'] = []
            log_to_textfile(self.path + '/Coach/' + 'logs.txt', 'Use uniform: ' + str(use_uniform) + '\n')
            
            # After MCTS is initialized, it requires the first time two expansaions
            tmp_num_expansions = self.num_expansions
            if (i==0) and (self.num_expansions==1):
                tmp_num_expansions = 2
            
            # Running search until full architecture is found
            while (len(s) // 5 != self.B):
                tmp_tmp_result = {}
                tmp_tmp_result['s'] = s
                
                # number of expansion for next move
                for j in range(tmp_num_expansions):
                    v, bl_trained, tmp_s = mcts.search(s,
                                                       ChildModel,
                                                       self.sess,
                                                       self.images, 
                                                       self.labels, 
                                                       self.path + '/Children/Insearchtrain/anychild_' + str(i) + '_' + str(j) + '_' + str(unique_idx) + '/', 
                                                       64,  
                                                       self.global_ops, 
                                                       self.global_param, 
                                                       'anychild_' + str(i) + '_' + str(j) + '_' + str(unique_idx),
                                                       self.variables_not_initialize,
                                                       self.GLOBAL_WEIGHTS,
                                                       self.max_noimprovements, 
                                                       self.max_iteration, 
                                                       self.lr_iteration_step, 
                                                       1, 
                                                       False,
                                                       lstm_perc,
                                                       lstm_epoch,
                                                       self.lstm_model,
                                                       use_uniform,
                                                       self.alphax_version,
                                                       self.debug_no_trainig
                                                      )
                    if bl_trained:
                        # If a ChildModel was trained in mcts.search
                        unique_idx = unique_idx + 1
                        log_to_textfile(self.path + '/Coach/' + 'logs.txt', 'In search iteration: ' + str(j) + ' v value: ' + str(v) + ' for ' + str(s) + ' \n')
                        convcell = s_to_convcell(tmp_s, self.B, self.action_space, self.combine_op)
                        log_to_textfile(self.path + '/Coach/' + 'logs.txt', 'Full state: ' + str(tmp_s) + ' convcell: ' + str(convcell) + ' \n')
                        if not(self.debug_no_trainig):
                            self.loader.save(self.sess, self.path + '/Coach/global_weights/global_weight')
                            self._reset_graph(False)
                        self.no_trained_in_search = self.no_trained_in_search + 1
                
                # Get next action based on MCTS probability
                probs = mcts.get_prob(s)
                # Keep track in replay history
                examples.append([s, probs, None, lstm_perc, lstm_epoch])
                action = np.random.choice(len(probs), p=probs)
                # Logging to dictionary
                tmp_tmp_nsa = {}
                for nsa_key in mcts.Nsa.keys():
                    if nsa_key[0] == s:
                        tmp_tmp_nsa[nsa_key] = mcts.Nsa[nsa_key]
                tmp_tmp_result['Nsa'] = tmp_tmp_nsa
                tmp_tmp_result['Ns'] = mcts.Ns[s]
                tmp_tmp_result['Ps'] = mcts.Ps[s]
                tmp_tmp_result['probs'] = probs
                tmp_tmp_result['action'] = action
                tmp123_result['in_search'].append(tmp_tmp_result)
                # Combine state with selected action
                s = s + (action,)
            
            tmp123_result['use_uniform'] = use_uniform
            pickle.dump(mcts, open(self.path + '/Coach/' + 'mcts' + str(no_mcts) + '.pickle', "wb"))
            log_to_textfile(self.path + '/Coach/' + 'logs.txt', 'Final action sequence: ' + str(s) + ' \n')
            convcell = s_to_convcell(s, self.B, self.action_space, self.combine_op)
            log_to_textfile(self.path + '/Coach/' + 'logs.txt', 'Final convcell: ' + str(convcell) + ' \n')
            createPath(self.path + '/Children/Finaltrain/finalchild_' + str(i))
            tmp123_result['final_seq'] = s
            tmp123_result['final_convcell'] = convcell
            
            # Train selected architecture for more epoch and report the accuracy
            if not(self.debug_no_trainig):
                model = ChildModel(self.sess,
                                   self.images, 
                                   self.labels, 
                                   self.path + '/Children/Finaltrain/finalchild_' + str(i) + '/', 
                                   64, 
                                   convcell, 
                                   self.global_ops, 
                                   self.global_param, 
                                   'finalchild_' + str(i),
                                   self.variables_not_initialize,
                                   self.no_channels_start)
                model.build_model(self.GLOBAL_WEIGHTS)
                model.couch_train(self.images, 
                                  self.labels,
                                  self.max_noimprovements, 
                                  self.max_iteration, 
                                  self.lr_iteration_step, 
                                  lstm_epoch, 
                                  self.no_global_variables)
                acc = model.predict_validation(self.images['valid'], self.labels['valid'], 0, initialize_new = False)
            else:
                acc = 0.2
            log_to_textfile(self.path + '/Coach/' + 'logs.txt', 'Final training validation accuracy: ' + str(acc) + ' \n')
            
            # Add the accuracy to the temporary replay history
            for e in examples:
                e[2] = acc
            
            self.no_trained_final = self.no_trained_final + 1
            self.loader.save(self.sess, self.path + '/Coach/global_weights/global_weight')
            self._reset_graph(False)
            tmp123_result['final_convcell_acc'] = acc
            tmp123_result['total_before_total_loss'] = 0
            tmp123_result['total_before_loss_value'] = 0
            tmp123_result['total_before_loss_prob'] = 0
            if i > 0:
                # Add temporary replay buffer to total replay buffer
                # Train LSTM
                while len(self.full_examples) > self.max_replay_size:
                    self.full_examples.pop(0)            
                self.full_examples = self.full_examples + examples

                total_before_total_loss = []
                total_before_loss_value = []
                total_before_loss_prob = []
                for k in range((len(self.full_examples) // self.lstm_batchsize) + 1):
                    before_total_loss, before_loss_value, before_loss_prob = self.lstm_model.train(self.full_examples, self.lstm_batchsize, self.lstm_learning_rate, self.path + '/Coach/' + 'logs.txt')
                    total_before_total_loss.append(before_total_loss)
                    total_before_loss_value.append(before_loss_value)
                    total_before_loss_prob.append(before_loss_prob)
                tmp123_result['total_before_total_loss'] = np.mean(total_before_total_loss)
                tmp123_result['total_before_loss_value'] = np.mean(total_before_loss_value)
                tmp123_result['total_before_loss_prob'] = np.mean(total_before_loss_prob)
            # Save dictory log files to disc
            total_results.append(tmp123_result)
            pickle.dump(self.full_examples, open(self.path + '/Coach/' + 'full_examples.pickle', "wb"))
            pickle.dump(total_results, open(self.path + '/Coach/' + 'total_results.pickle', "wb"))
        self.mcts = mcts