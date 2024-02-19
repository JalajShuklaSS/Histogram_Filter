import numpy as np


class HistogramFilter(object):
    """
    Class HistogramFilter implements the Bayes Filter on a discretized grid space.
    """

    def histogram_filter(self, cmap, belief, action, observation):
        '''
        Takes in a prior belief distribution, a colormap, action, and observation, and returns the posterior
        belief distribution according to the Bayes Filter.
        :param cmap: The binary NxM colormap known to the robot.
        :param belief: An NxM numpy ndarray representing the prior belief.
        :param action: The action as a numpy ndarray. [(1, 0), (-1, 0), (0, 1), (0, -1)]
        :param observation: The observation from the color sensor. [0 or 1].
        :return: The posterior distribution.
        '''

        ### Your Algorithm goes Below.
        # declaring the constant values
        P_movement = 0.9
        P_stayed = 0.1
        P_reading_given_color = 0.9
        P_fail_color = 0.1
        cmap = np.rot90(cmap, -1)
        belief = np.rot90(belief, -1)
        #set uniform prior belief
        rows, cols = cmap.shape
        prior_belief = belief
        
        alpha = np.zeros_like(prior_belief)
        eta = 0
        
        #transition 
        for i in range(rows):
            for j in range(cols):
                    k = action
                    moved_i, moved_j = i+k[0], j+k[1]
                    if moved_i >= 0 and moved_i < rows and moved_j >= 0 and moved_j < cols:
                        alpha[moved_i, moved_j] += prior_belief[i, j] * P_movement
                        alpha[i, j] += prior_belief[i, j] * P_stayed
                    else:
                        alpha[i, j] += prior_belief[i, j] * 1
        
        #observation 
        for i in range(rows):
            for j in range(cols):
                if cmap[i, j] == observation:
                    alpha[i, j] *= P_reading_given_color
                else:
                    alpha[i, j] *= P_fail_color
        
        #normalizing
        eta = 1 / np.sum(alpha)
        
        norm_belif = alpha * eta

        return np.rot90(norm_belif, 1)
    
        
        
        
        
      
        
    
        
        
