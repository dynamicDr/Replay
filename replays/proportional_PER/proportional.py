import numpy
import random

import numpy as np

from . import sum_tree
from replays.base_replay import BaseReplay


class ProportionalPER(BaseReplay):
    """ The class represents prioritized experience replay buffer.

    The class has functions: store samples, pick samples with 
    probability in proportion to sample's priority, update 
    each sample's priority, reset alpha.

    see https://arxiv.org/pdf/1511.05952.pdf .

    """
    
    def __init__(self,max_size,batch_size,alpha=0.7,beta = 0.7):
        """ Prioritized experience replay buffer initialization.
        
        Parameters
        ----------
        max_size : int
            sample size to be stored
        batch_size : int
            batch size to be selected by `select` method
        alpha: float
            exponent determine how much prioritization.
            Prob_i \sim priority_i**alpha/sum(priority**alpha)
        """
        super().__init__(max_size,batch_size)
        self.tree = sum_tree.SumTree(self.max_size)
        self.alpha = alpha
        self.beta = beta

    def add(self, data, priority):
        """ Add new sample.
        
        Parameters
        ----------
        data : object
            new sample
        priority : float
            sample's priority
        """
        self.tree.add(data, priority**self.alpha)
        self.size = self.tree.size

    def sample(self):
        """ The method return samples randomly.
        
        Parameters
        ----------
        beta : float
        
        Returns
        -------
        out : 
            list of samples
        weights: 
            list of weight
        indices:
            list of sample indices
            The indices indicate sample positions in a sum tree.
        """
        
        if self.tree.filled_size() < self.batch_size:
            return None, None, None

        indices = []
        weights = []
        priorities = []
        state, action, reward, next_state, done = [], [], [], [], []
        for _ in range(self.batch_size):
            r = random.random()
            data, priority, index = self.tree.find(r)
            priorities.append(priority)
            weights.append((1./self.max_size/priority)**self.beta if priority > 1e-16 else 0)
            indices.append(index)
            self.priority_update([index], [0]) # To avoid duplicating
            s, a, r, s_, d = data
            state.append(np.array(s, copy=False))
            action.append(np.array(a, copy=False))
            reward.append(np.array(r, copy=False))
            next_state.append(np.array(s_, copy=False))
            done.append(np.array(d, copy=False))

        self.priority_update(indices, priorities) # Revert priorities

        # Normalize for stability
        max_weight = max(weights)
        if max_weight!=0:
            for i in range(len(weights)):
                weights[i] /= max_weight
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done), weights, indices

    def priority_update(self, indices, priorities):
        """ The methods update samples's priority.

        Parameters
        ----------
        indices :
            list of sample indices
        """
        for i, p in zip(indices, priorities):
            self.tree.val_update(i, p**self.alpha)

    # def reset_alpha(self, alpha):
    #     """ Reset a exponent alpha.
    #
    #     Parameters
    #     ----------
    #     alpha : float
    #     """
    #     self.alpha, old_alpha = alpha, self.alpha
    #     priorities = [self.tree.get_val(i)**-old_alpha for i in range(self.tree.filled_size())]
    #     self.priority_update(range(self.tree.filled_size()), priorities)

        
            
        
        
