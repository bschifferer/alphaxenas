import math
import numpy as np

from alphaxenas.model_utils import legal_action_from_seq, one_hot, s_to_convcell, log_to_textfile
from alphaxenas.data_utils import createPath

# Constant for avioding devision by zero
EPS = 1e-8

class MCTS():
    """
    Monte Carlo Tree Search (MCTS) class - guides the search through space based on MCTS
    
    Reference: https://web.stanford.edu/~surag/posts/alphazero.html
    
    Functions:
        get_prob (list): Returns the action probability based on how MCTS explored different actions
        search (tuple(int, boolean, tuple): Explores an action (new or old) given a state

    Attributes:
        See incomment of __init__ function
    """
    def __init__(self, action_size, B, num_ops, num_combine, action_space, combine_op, filename, no_channels_start):
        """
        Function __init__

        Initialize the MCTS tree - save paramaters as attributes

        Args:
            action_size (int): # of possible actions
            B (int): # of incell cells
            num_ops (int): # of possible neural network operation
            num_combine (int): # of possible combine operation
            action_space (dict): Dictionary with possible neural network operations + weights
            combine_op (dict): Dictionary with possible combine operations + weights
            filename (str): Filename of Coachs log
            no_channels_start (int): Number of channels in convolution layer
        
        Attributes:
            Qsa (dict): Stores Q value for a state, action pair
            Nsa (dict): Stores # of visits of a state, action pair
            Ns (dict): Stores # of visits of a state
            Ps (dict): Stores initial probability of actions given a state
            Es (dict): Stores endreward (accuracy) of state s
            Vs (dict): Stores valid actions in state s
        """
        self.action_size = action_size
        self.B = B
        self.num_ops = num_ops
        self.num_combine = num_combine
        self.action_space = action_space
        self.combine_op = combine_op
        self.Qsa = {}
        self.Nsa = {}
        self.Ns = {}
        self.Ps = {}
        self.Es = {}
        self.Vs = {}
        self.filename = filename
        self.no_channels_start = no_channels_start

    def get_prob(self, state):
        """
        Function get_prob

        return the action probability explored by the MCTS tree

        Args:
            state (tuple): a tuple of actions which defines the state
        
        Return:
            probs (list): Return the action probabilties for state s 
        """
        s = state
        counts = [self.Nsa[(s,a)] if (s,a) in self.Nsa else 0 for a in range(self.action_size)]
        probs = [x/float(sum(counts)) for x in counts]
        return(probs)
    
    def search(self, 
               state, 
               ChildModel, 
               sess, 
               images, 
               labels, 
               child_output, 
               batch_size,  
               global_ops, 
               global_param, 
               childname,
               variables_not_initialize,
               GLOBAL_WEIGHTS,
               max_noimprovements,
               max_iteration,
               lr_iteration_step,
               max_epochs,
               no_global_variables,
               lstm_perc,
               lstm_epoch, 
               lstm_model, 
               use_uniform,
               alphax_version,
               debug_no_trainig):
        """
        Function search

        Executes a action search for a given state based on MCTS and updates all values 

        Args:
            state (tuple): starting state for MCTS search
            ChildModel (class ChildModel): class of ChildModel used for training a full state
            sess (tensorflow session): Tensorflow session used for training ChildModel
            images (dict): Images used for training ChildModel
            labels (dict): Labels used for training ChildModel
            child_output (str): Child output path 
            batch_size (int): Batchsize for training child  
            global_ops (list): Not used anymore 
            global_param (list): List with some global parameters
            childname (str): Name of ChildModel
            variables_not_initialize (list): List of global variables, which should not be initialized
            GLOBAL_WEIGHTS (dict): Dictonary with shared tensorflow weights
            max_noimprovements (int): Stop training ChildModel if does not improve over number of epochs
            max_iteration (int): Number of maximal training steps in ChildModel
            lr_iteration_step (list): Epochs when learning rate is decayed in ChildModel
            max_epochs (int): Number of epochs for training ChildModel
            no_global_variables (boolean): If True, no global variables are trained in ChildModel
            lstm_perc (int): Percentage of data used for child model
            lstm_epoch (int): Number of epochs for child model
            lstm_model (object of class LSTMModel): LSTM model for prediction v, Ps
            use_uniform (boolean): If True uniform distribution is used for initializing Ps
            alphax_version (boolean): If True other MCTS formula is used (based on AlphaX paper)
            debug_no_trainig (boolean): If True no ChildModel are trained and only dummy value is returned
        
        Return: v, bl_trained, tmp_s
            _ (float): Value of requested state s
            _ (boolean): If True, then a new ChildModel was trained
            _ (tuple): Selected action
        """
        s = state
        
        if len(s) // 5 == self.B:
            # Full state sequence is discovered
            if s not in self.Es:
                # If s hasn't been discovered yet - train a ChildModel
                convcell = s_to_convcell(s, self.B, self.action_space, self.combine_op)
                createPath(child_output)
                log_to_textfile(self.filename, 'MCTS State: ' + str(s) + ' MCTS cell: ' + str(convcell) + '\n')
                if debug_no_trainig:
                    return(0.2, True, s)
                model = ChildModel(sess,
                                   images, 
                                   labels, 
                                   child_output, 
                                   batch_size, 
                                   convcell, 
                                   global_ops, 
                                   global_param, 
                                   childname,
                                   variables_not_initialize, 
                                   self.no_channels_start)
                model.build_model(GLOBAL_WEIGHTS)
                model.couch_train(images, 
                                  labels,
                                  max_noimprovements, 
                                  max_iteration, 
                                  lr_iteration_step, 
                                  max_epochs, 
                                  no_global_variables)
                acc = model.predict_validation(images['valid'], labels['valid'], 0, initialize_new = False)
                self.Es[s] = acc
                return acc, True, s
            if self.Es[s]!=0:
                # s was already discovered
                return self.Es[s], False, s

        if s not in self.Ps:
            # Initialize Policy by neural network
            ps, v = lstm_model.pred_action_sequence([s], [lstm_perc], [lstm_epoch])
            ps = ps[0]
            v = v[0]
            # Use uniform distribution instead of LSTM
            if use_uniform:
                ps = np.asarray(ps)
                ps = np.ones_like(ps)
            self.Ps[s] = ps
            # Only valid actions
            valids = legal_action_from_seq(s, self.B, self.num_ops, self.num_combine, self.action_size)
            valids_hotn = np.sum(one_hot(np.asarray(valids), self.action_size), axis=0)
            self.Ps[s] = self.Ps[s]*valids_hotn
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                # Normalize it again
                self.Ps[s] /= sum_Ps_s
            else:
                print("All valid moves were masked, do workaround.")
            self.Vs[s] = valids_hotn
            self.Ns[s] = 0
            return v, False, s

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # Select the action with the highest upper confidence bound based on MCTS formula
        for a in range(self.action_size):
            if valids[a]:
                if (s,a) in self.Qsa:
                    if alphax_version:
                        # Use AlphaX rule
                        u = self.Qsa[(s,a)]/self.Nsa[(s,a)] + 2*200*math.sqrt(2*math.log10(self.Ns[s])/(1+self.Nsa[(s,a)]))
                    else:
                        # Use AlphaGo rule
                        u = self.Qsa[(s,a)] + 5*self.Ps[s][a]*math.sqrt(self.Ns[s])/(1+self.Nsa[(s,a)])
                else:
                    if alphax_version:
                        # Use AlphaX rule
                        u = 0/self.Nsa[(s,a)] + 2*200*math.sqrt(2*math.log10(self.Ns[s])/(1+self.Nsa[(s,a)]))
                    else:
                        # Use AlphaGo rule
                        u = 5*self.Ps[s][a]*math.sqrt(self.Ns[s] + EPS)     # Q = 0 ?
                if u > cur_best:
                    # Keep best action
                    cur_best = u
                    best_act = a

        a = best_act
        next_s = s + (a,)
        # Find value for best action a in state s
        v, bl_trained, tmp_s = self.search(next_s,
                                           ChildModel, 
                                           sess, 
                                           images, 
                                           labels, 
                                           child_output, 
                                           batch_size,  
                                           global_ops, 
                                           global_param, 
                                           childname,
                                           variables_not_initialize,
                                           GLOBAL_WEIGHTS,
                                           max_noimprovements,
                                           max_iteration,
                                           lr_iteration_step,
                                           max_epochs,
                                           no_global_variables,
                                           lstm_perc,
                                           lstm_epoch, 
                                           lstm_model,
                                           use_uniform,
                                           alphax_version,
                                           debug_no_trainig
                                          )

        # Update Q value of state action pair (s,a)
        if (s,a) in self.Qsa:
            if alphax_version:
                self.Qsa[(s,a)] = self.Qsa[(s,a)] + v
            else:
                self.Qsa[(s,a)] = (self.Nsa[(s,a)]*self.Qsa[(s,a)] + v)/(self.Nsa[(s,a)]+1)
            self.Nsa[(s,a)] += 1

        else:
            self.Qsa[(s,a)] = v
            self.Nsa[(s,a)] = 1

        self.Ns[s] += 1
        return v, bl_trained, tmp_s