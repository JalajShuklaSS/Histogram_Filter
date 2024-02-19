import numpy as np
import matplotlib.pyplot as plt
from histogram_filter import HistogramFilter
import random


if __name__ == "__main__":

    # Load the data
    data = np.load(open('C:/D/2024/Spring Sem courses/ESE 650 Learning in robo/hw1/starter.npz', 'rb'))
    cmap = data['arr_0'] #nxm =
    actions = data['arr_1'] #kx2
    observations = data['arr_2'] #k
    belief_states = data['arr_3'] #kxnxm

    new_belief = belief_states[0]
    print("belief_states: \n", belief_states)
    print(belief_states.shape)
    
    hf =HistogramFilter()
    new_belief = 1 / 400 * np.ones((20, 20))

    for i in range(30):
    

        # Update new_belief using the current action and observation
        new_belief = hf.histogram_filter(cmap, new_belief, actions[i,:], observations[i])
       
    print(new_belief.shape)
        # Plot the new_belief
    plt.imshow(new_belief, cmap='gray')
    plt.show()
    plt.pause(0.1)
    plt.clf()
    plt.close()
        