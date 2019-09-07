import numpy as np
import tensorflow as tf

from alphaxenas.model_utils import log_to_textfile

class LSTMModel():
    """
    LSTM Model - guides the MCTS search by predicting the action probabilities and value for a given state
    
    Functions:
        one_hot (a, num_classes): Return a one-hot encoded array given an action index and the maximum number of classers
        prepare_action_sequence (seq_action, seq_perc, seq_epoch): Given input parameters, prepare the arries to feed to LSTM model
        pred_action_sequence (seq_action, seq_perc, seq_epoch): Given input parameters, predict the action distribution and value
        train (full_examples, batch_size, learning_rate, coach_filename): Train the LSTM model
        _build_model (state_size, action_size, lstm_size, B, seq_max_len): Build the LSTM model

    Attributes:
        See incomment of __init__ function
    """
    def __init__(self, sess, state_size, action_size, lstm_size, B, path):
        """
        Function __init__

        Initialize the LSTM model - save paramaters as attributes

        Args:
            sess (Tensorflow session): Tensorflow session, where the parameters are saved
            state_size (int): Size of input state (# of different actions)
            action_size (int): Size of output state (# of different actions)
            size_lstm (int): Size of hidden layer in LSTM
            B (int): Number of inconv cells
            path (str): Directory to store LSTM files
        
        Attributes:
            seq_max_len (int): Maximal sequence length (number of cells * 5 parameter per cell)
        """
        self.sess = sess
        self.state_size = state_size
        self.action_size = action_size
        self.lstm_size = lstm_size
        self.B = B
        self.seq_max_len = B*5
        self.path = path
        self._build_model(state_size, action_size, lstm_size, B, B*5)
    
    def one_hot(self, a, num_classes):
        """
        Function one_hot

        Return a one-hot encoded array given an action index and the maximum number of classers

        Args:
            a (int): Index of action
            num_classes (int): Maximal number of different actions
            
        Return:
            _ (numpy array): Numpy array with one-hot encoded vector
        """
        return np.squeeze(np.eye(num_classes)[a.reshape(-1)])
    
    def prepare_action_sequence(self, seq_action, seq_perc, seq_epoch):
        """
        Function prepare_action_sequence

        Given input parameters, prepare the arries to feed to LSTM model

        Args:
            seq_action (list): List of action sequences
            seq_perc (float): Percentage used of training dataset
            seq_epoch (int): Number of epochs used for training
            
        Return:
            x_batch (numpy array): Numpy array of data, which can fed as input data to LSTM 
            batch_size (int): Number of sequences
            seq_len (list): List with entries how long each sequence is
        """
        seq_len = [len(x) for x in seq_action]
        batch_size = len(seq_action)
        x_batch = np.zeros([batch_size, self.seq_max_len, self.state_size+2])
        for i in range(len(seq_action)):
            for j in range(len(seq_action[i])):
                x_batch[i,j,:] = x_batch[i,j,:] = np.hstack([self.one_hot(np.asarray(seq_action[i][j]), self.action_size).astype(int), seq_perc[i], seq_epoch[i]])
        return(x_batch, batch_size, seq_len)
    
    def pred_action_sequence(self, seq_action, seq_perc, seq_epoch):
        """
        Function pred_action_sequence

        Given input parameters, predict the action distribution and value

        Args:
            seq_action (list): List of action sequences
            seq_perc (float): Percentage used of training dataset
            seq_epoch (int): Number of epochs used for training
            
        Return:
            nn_out[0] (numpy array): Numpy array with probabilities distribution for next action for each prediction
            nn_out[0] (numpy array): Numpy array with state value for each prediction
        """
        x_batch, batch_size, seq_len = self.prepare_action_sequence(seq_action, seq_perc, seq_epoch)
        nn_out = self.sess.run([self.prob, 
                                self.value, 
                                self.cell_state], 
                               feed_dict={self.x:x_batch, self.batch_size:batch_size, self.seq_len: seq_len})
        return(nn_out[0], nn_out[1])
    
    def train(self, full_examples, batch_size, learning_rate, coach_filename):
        """
        Function train

        Train the LSTM model (one training step)

        Args:
            full_examples (list): Replay history
            batch_size (int): Batch size for LSTM to train
            learning_rate (float): Learning rate for LSTM model 
            coach_filename (str): Filename of Coach model to log results
            
        Return:
            before_total_loss (float): Total loss for the current batch
            before_loss_value  (float): Loss for state value for the current batch
            before_loss_prob  (float): Loss for the probability distribution for the current batch
        """
        idx = np.asarray(range(len(full_examples)))
        np.random.shuffle(idx)
        idx = idx[range(min(batch_size, idx.shape[0]))]
        states = []
        probs = []
        acc = []
        perc = []
        epoch = []
        for i in idx:
            states.append(full_examples[i][0])
            probs.append(full_examples[i][1])
            acc.append(full_examples[i][2])
            perc.append(full_examples[i][3])
            epoch.append(full_examples[i][4])
        
        acc = np.asarray(acc).reshape(-1,1)
        x_batch, tmp_batch_size, seq_len = self.prepare_action_sequence(states, perc, epoch)
        before_total_loss, before_loss_value, before_loss_prob  = self.sess.run([self.total_loss, 
                                                                                 self.loss_value, 
                                                                                 self.loss_prob], feed_dict={self.x: x_batch, 
                                                                                                             self.target_pis: probs, 
                                                                                                             self.target_v: acc,
                                                                                                             self.seq_len: seq_len,
                                                                                                             self.batch_size: tmp_batch_size,
                                                                                                             self.learning_rate: learning_rate})
        
        _, total_loss, loss_value, loss_prob  = self.sess.run([self.train_step, 
                                                               self.total_loss, 
                                                               self.loss_value, 
                                                               self.loss_prob], feed_dict={self.x: x_batch, 
                                                                                           self.target_pis: probs, 
                                                                                           self.target_v: acc,
                                                                                           self.seq_len: seq_len,
                                                                                           self.batch_size: tmp_batch_size,
                                                                                           self.learning_rate: learning_rate
                                                                                          })
        log_to_textfile(self.path + 'logs.txt', 'B LSTM Total loss: ' + str(before_total_loss) + ' B Loss prob: ' + str(before_loss_prob) + ' B Loss value: ' + str(before_loss_value) + '\n')
        log_to_textfile(self.path + 'logs.txt', 'LSTM Total loss: ' + str(total_loss) + ' Loss prob: ' + str(loss_prob) + ' Loss value: ' + str(loss_value) + '\n')
        log_to_textfile(coach_filename, 'B LSTM Total loss: ' + str(before_total_loss) + ' B Loss prob: ' + str(before_loss_prob) + ' B Loss value: ' + str(before_loss_value) + '\n')
        log_to_textfile(coach_filename, 'LSTM Total loss: ' + str(total_loss) + ' Loss prob: ' + str(loss_prob) + ' Loss value: ' + str(loss_value) + '\n')
        return(before_total_loss, before_loss_value, before_loss_prob)
        
    
    def _build_model(self, state_size, action_size, lstm_size, B, seq_max_len):
        """
        Function _build_model

        Build the LSTM model

        Args:
            state_size (int): Size of input state (# of different actions)
            action_size (int): Size of output state (# of different actions)
            lstm_size (int): Size of hidden layer in LSTM
            B (int): Number of inconv cells
            seq_max_len (int): List with entries how long each sequence is
        """
        with tf.variable_scope('controller_lstm_'):
            learning_rate = tf.placeholder(tf.float32, shape=[])
            x = tf.placeholder(tf.float32, [None, seq_max_len, state_size+2], name='input_placeholder')
            target_pis = tf.placeholder(tf.float32, [None, action_size], name='pis_placeholder')
            target_v = tf.placeholder(tf.float32, [None, 1], name='v_placeholder')
            batch_size = tf.placeholder(tf.int32, [], name='batch_size')
            seq_len = tf.placeholder(tf.int32, [None])
            
            #cell = tf.nn.rnn_cell.BasicRNNCell(lstm_size)
            cell = tf.nn.rnn_cell.LSTMCell(lstm_size)
            outputs, states = tf.nn.dynamic_rnn(cell, x, sequence_length=seq_len, dtype=tf.float32)
            
            # Hack to build the indexing and retrieve the right output.
            tmp_batch_size = tf.shape(outputs)[0]
            index = tf.range(0, tmp_batch_size) * seq_max_len + (seq_len - 1)
            
            lstm_output = tf.gather(tf.reshape(outputs, [-1, lstm_size]), index)
            
            with tf.variable_scope('prob_softmax'):
                prob_W = tf.get_variable('W', [lstm_size, action_size])
                prob_b = tf.get_variable('b', [action_size], initializer=tf.constant_initializer(0.0))
            with tf.variable_scope('value'):
                value_W = tf.get_variable('W', [lstm_size, 1])
                value_b = tf.get_variable('b', [1], initializer=tf.constant_initializer(0.0))
            
            prob = tf.add(tf.nn.softmax(tf.matmul(lstm_output, prob_W) + prob_b), tf.constant(0.0001))
            value = tf.sigmoid(tf.matmul(lstm_output, value_W) + value_b)
            target_pis_norm = tf.add(target_pis, tf.constant(0.0001))
            
            p_dist = tf.distributions.Categorical(probs=prob)
            q_dist = tf.distributions.Categorical(probs=target_pis_norm)
            
            #loss_prob = -tf.reduce_sum(prob * target_pis)
            loss_prob = tf.reduce_mean(tf.distributions.kl_divergence(p_dist, q_dist, allow_nan_stats=True))
            loss_value = tf.losses.mean_squared_error(predictions=value, labels=target_v)
            total_loss = loss_prob+loss_value
            
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)
            
            self.learning_rate = learning_rate
            self.x = x
            self.target_pis = target_pis
            self.target_v = target_v
            self.batch_size = batch_size
            self.seq_len = seq_len
            self.lstm = cell
            self.cell_state = states
            self.total_loss = total_loss
            self.loss_value = loss_value
            self.loss_prob = loss_prob
            self.prob = prob
            self.value = value
            self.train_step = train_step