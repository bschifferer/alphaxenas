import tensorflow as tf
import numpy as np

import pickle
import matplotlib.pyplot as plt

def get_index_best_archteicture(total_result):
    """
    Function get_index_best_archteicture
    
    Get index best architecture based on final accuracy back
    
    Args:
        total_result (list): Pickle object which is saved by Coach class 
    
    Return:
        _ (int): Index of the run with the best accuracy
    """
    acc = []
    for result in total_result:
        acc.append(result['final_convcell_acc'])
    return(np.argmax(acc))

def print_diff_architectures_to_jupyter(total_result):
    """
    Function print_diff_architectures_to_jupyter
    
    Print the final accuracy per iteration to console/jupyter notebook
    
    Args:
        total_result (list): Pickle object which is saved by Coach class 
    
    Return:
    
    """
    for result in total_result:
        print('################ New Final Train ################')
        print('Architecture')
        print(result['final_convcell'])
        print('Accuracy ' + str(result['final_convcell_acc']))

def plot_mcts_resuts(total_result):
    """
    Function plot_mcts_resuts
    
    Plot different mectrics from the pickle object saved by Coach class
    
    Args:
        total_result (list): Pickle object which is saved by Coach class 
    
    Return:
    
    """
    i = []
    final_acc = []
    lstm_total_loss = []
    lstm_prob_loss = []
    lstm_v_loss = []
    for result in total_result:
        i.append(result['i'])
        final_acc.append(result['final_convcell_acc'])
        lstm_total_loss.append(result['total_before_total_loss'])
        lstm_prob_loss.append(result['total_before_loss_prob'])
        lstm_v_loss.append(result['total_before_loss_value'])
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15,8))
    axes[0, 0].plot(i, final_acc)
    axes[0, 0].set_title('Child accuracy after each MCTS iteration')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 1].plot(i, lstm_total_loss)
    axes[0, 1].set_title('LSTM total loss after each MCTS iteration')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Loss')
    axes[1, 0].plot(i, lstm_prob_loss)
    axes[1, 0].set_title('LSTM Prob loss after each MCTS iteration')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 1].plot(i, lstm_v_loss)
    axes[1, 1].set_title('LSTM Value loss after each MCTS iteration')
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Loss')
    fig.subplots_adjust(hspace=.5)
    plt.show()

def load_pickle(filename):
    """
    Function load_pickle
    
    load a pickle file based on a file
    
    Args:
        filename (str): Filename
    
    Return:
        _ (<object>): Return the saved object in the pickle file
    """
    with open(filename, 'rb') as f:
        x = pickle.load(f)
    return(x)

def print_files_to_jupyter(filename):
    """
    Function print_files_to_jupyter
    
    Print the logfiles to console/jupyter notebook line-by-line
    
    Args:
        filename (str): Filename of logfile
    
    Return:
    
    """
    f = open(filename)
    line = f.readline()
    while line:
        print(line)
        line = f.readline()
    f.close()

def log_to_textfile(filename, text):
    """
    Function log_to_textfile
    
    Appends a text to a file (logs)
    
    Args:
        filename (str): Filename of logfile
        text (str): New information to log (append)
    
    Return:
    
    """
    f = open(filename, "a")
    f.write(text)
    f.close()

def legal_action(idx_b, idx_incov, B, num_ops, num_combine, num_actions):
    """
    Function legal_action
    
    Return the possible legal action (operation)
    
    Args:
        idx_b (int): Current index in the convolution cell 
        idx_incov (int):  Current index of operation to predict
        B (int): Number of inconv cells
        num_ops (int): Number of different operation (convolution, pooling, id)  
        num_combine (int): Number of different combine operation (concat, add)
        num_actions (int): Total possible number of actions
        
    Return:
        mask (numpy array): Return the mask (0 = illegal action, 1 = legal action)
    """
    mask = np.zeros(num_actions)
    illagel_mask = np.ones(num_actions, dtype=bool)
    if idx_incov in [0, 1]:
        illagel_mask[np.asarray(range(idx_b-1+1))] = False
    if idx_incov in [2, 3]:
        illagel_mask[np.asarray(range(B,B+num_ops))] = False
    if idx_incov in [4]:
        illagel_mask[np.asarray(range(B+num_ops,B+num_ops+num_combine))] = False
    mask[illagel_mask] = -9999999
    return(mask)

def legal_action_from_seq(s, B, num_ops, num_combine, num_actions):
    """
    Function legal_action_from_seq
    
    Return the possible legal action in current state / sequence s
    
    Args:
        s (tuple): Current state/action sequence
        B (int): Number of inconv cells
        num_ops (int): Number of different operation (convolution, pooling, id)  
        num_combine (int): Number of different combine operation (concat, add)
        num_actions (int): Total possible number of actions
        
    Return:
        _ (list): List of legal action indices
    """
    num_prev_action = len(s)
    idx_b = num_prev_action // 5
    idx_incov = num_prev_action - idx_b*5
    mask = legal_action(idx_b+1, idx_incov, B, num_ops, num_combine, num_actions)
    ind_legal_moves = np.where(mask != -9999999)[0]
    return(list(ind_legal_moves))

def one_hot(a, num_classes):
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

def s_to_convcell(s, B, action_space, combine_op):
    """
    Function s_to_convcell
    
    Returns a sequence of action indices (s) to a list of convolutional cells
    Example:
        s = (0, 0, 7, 9, 14, 0, 1, 12, 8, 14, 0, 2, 11, 4, 15, 0, 0, 5, 4, 14)
        convcell = [[0, 0, 'conv_1x1', 'convsep_3x3', 'add'], 
                    [0, 1, 'maxpool_3x3', 'id_', 'add'], 
                    [0, 2, 'convsep_7x7', 'conv_3x3', 'concat'], 
                    [0, 0, 'conv_5x5', 'conv_3x3', 'add']]
    
    Args:
        s (tuple): Current state/action sequence
        B (int): Number of inconv cells 
        action_space (list): List of different operation (convolution, pooling, id)  
        combine_op (list): List of different combine operation (concat, add)
            
    Return:
        Output (list): A list of operation for the convolutional cell
    """
    output = []
    cell = []
    counter_cell = 0
    for el in s:
        if counter_cell in [0, 1]:
            cell.append(el)
        if counter_cell in [2, 3]:
            cell.append([action_space[x]['name'] for x in action_space.keys()][el-B])
        if counter_cell in [4]:
            cell.append([combine_op[x] for x in combine_op.keys()][el-B-len(action_space.keys())])
        counter_cell += 1
        if counter_cell == 5:
            counter_cell = 0
            output.append(cell)
            cell = []
    return(output)

def get_weights(name, shape):
    """
    Function get_weights
    
    Return tensorflow weights given the shape
    
    Args:
        name (str): Name of the weight
        shape (list): Shape of the weight
            
    Return:
        _ (tensorflow weights): A tensorflow weight
    """
    return(tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer(), trainable=True))


def create_weights(N_numberofconv, N_convcells, B, action_space, no_channels_start):
    """
    Function get_weights
    
    Return tensorflow weights given the shape
    
    Args:
        N_numberofconv (int): Number of times to repeat convolutional cells (kept to 1)
        N_convcells (int): Number of convolutional cells
        B (int): Number of inconv cells 
        action_space (list): List of different operation (convolution, pooling, id)  
        no_channels_start (int): Number of default filter size
            
    Return:
        output (dict): A dictionary wight tensorflow weights
    """
    output = {}
    for i in range(N_numberofconv):
        output[i] = {}
        for j in range(N_convcells):
            output[i][j] = {}
            for k in range(B):
                output[i][j][k] = {}
                for l in action_space.keys():
                    if action_space[l]['type'] == 'conv':
                        output[i][j][k][action_space[l]['name']] = [get_weights(str(i) + '_' +
                                                                               str(j) + '_' +
                                                                               str(k) + '_' +
                                                                               action_space[l]['name'] + 'small', 
                                                                            [action_space[l]['shape'][0], 
                                                                             action_space[l]['shape'][1], 
                                                                             no_channels_start, 
                                                                             no_channels_start]),
                                                                    get_weights(str(i) + '_' +
                                                                               str(j) + '_' +
                                                                               str(k) + '_' +
                                                                               action_space[l]['name'] + 'big', 
                                                                            [action_space[l]['shape'][0], 
                                                                             action_space[l]['shape'][1], 
                                                                             2*no_channels_start, 
                                                                             no_channels_start])
                                                                   ]
                    if action_space[l]['type'] == 'convsep':
                        output[i][j][k][action_space[l]['name']] = [get_weights(str(i) + '_' +
                                                                                str(j) + '_' +
                                                                                str(k) + '_' +
                                                                                action_space[l]['name'] + 'small_depthwise', 
                                                                                [action_space[l]['shape'][0], 
                                                                                 action_space[l]['shape'][1], 
                                                                                 no_channels_start, 
                                                                                 1]),
                                                                    get_weights(str(i) + '_' +
                                                                                str(j) + '_' +
                                                                                str(k) + '_' +
                                                                                action_space[l]['name'] + 'small_pointwise', 
                                                                                [1, 
                                                                                 1, 
                                                                                 no_channels_start, 
                                                                                 no_channels_start]),
                                                                    get_weights(str(i) + '_' +
                                                                                str(j) + '_' +
                                                                                str(k) + '_' +
                                                                                action_space[l]['name'] + 'big_depthwise', 
                                                                                [action_space[l]['shape'][0], 
                                                                                 action_space[l]['shape'][1], 
                                                                                 2*no_channels_start, 
                                                                                 1]),
                                                                    get_weights(str(i) + '_' +
                                                                                str(j) + '_' +
                                                                                str(k) + '_' +
                                                                                action_space[l]['name'] + 'big_pointwise', 
                                                                                [1, 
                                                                                 1, 
                                                                                 2*no_channels_start, 
                                                                                 no_channels_start])]
    return(output)