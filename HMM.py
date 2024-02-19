import numpy as np

class HMM():

    def __init__(self, Observations, Transition, Emission, Initial_distribution):
        self.Observations = Observations
        self.Transition = Transition
        self.Emission = Emission
        self.Initial_distribution = Initial_distribution

    def forward(self):
        obs = len(self.Observations)
        states = self.Transition.shape[0]
        
        alpha = np.zeros((obs, states))
        alpha[0, :] = self.Initial_distribution * self.Emission[:, self.Observations[0]]
      
        for i in range(1, obs):
            for j in range(states):
                alpha[i, j] = np.dot(alpha[i - 1, :], self.Transition[:, j]) * self.Emission[j, self.Observations[i]]
        
            # alpha[i, :] = alpha[i, :] / np.sum(alpha[i, :])
        return alpha

    def backward(self):

        obs = len(self.Observations)
        states = self.Transition.shape[0]
        
        beta = np.zeros((obs, states))
        beta[-1, :] = np.array([1, 1])
        
        for i in range(obs - 2, -1, -1):
            for j in range(states):
                beta[i, j] = np.dot(self.Transition[j, :], self.Emission[:, self.Observations[i + 1]] * beta[i + 1, :])

        
        return beta

    def gamma_comp(self, alpha, beta):

        normalization = np.sum (alpha[-1])
        gamma1 = alpha * beta
        gamma  = gamma1 / normalization

        return gamma

    def xi_comp(self, alpha, beta, gamma):
        
        obs = len(self.Observations)
        states = self.Transition.shape[0]
        
        xi = np.zeros((obs - 1, states, states))
        for i in range(obs - 1):
            for j in range(states):
                for k in range(states):
                    xi[i, j, k] = alpha[i, j] * self.Transition[j, k] * self.Emission[k, self.Observations[i + 1]] * beta[i + 1, k]
            xi[i, :, :] /= np.sum(xi[i, :, :])
        
        return xi

    def update(self, alpha, beta, gamma, xi):
        X = self.Transition.shape[0]
        E = self.Emission.shape[1]

        new_init_state = gamma[0]


        sum_xi = np.sum(xi, axis=0)
        sum_gamma1 = np.sum(gamma[:-1], axis=0)
        sum_gamma = sum_gamma1[:, np.newaxis]
        Tp = sum_xi / sum_gamma
        Mp = np.zeros_like(self.Emission)
        Mp = (gamma.T @ np.eye(E)[self.Observations]) / np.sum(gamma, axis=0).reshape(X, 1)
        
        M_prime = Mp
        T_prime = Tp

        return T_prime, M_prime, new_init_state

    def trajectory_probability(self, alpha, beta, T_prime, M_prime, new_init_state):

        P_original = np.array([0.])
        P_prime = np.array([0.])
        
        O = self.Observations.shape[0]
        
        P_original = np.sum(alpha[O - 1, :])
        
        A_1 = new_init_state * M_prime[:, self.Observations[0]]
        A_n_t = A_1.T
        
        for i in range(O - 1):
            A_1 = M_prime[:, self.Observations[i + 1]] * (A_1.T @ T_prime)
            A_n_t = np.vstack((A_n_t, A_1.T))      
        PP = np.sum(A_n_t[O - 1, :])
        
        P_prime = PP

        return P_original, P_prime


if __name__ == "__main__":

    # Load the data
    data= np.load(open('C:/D/2024/Spring Sem courses/ESE 650 Learning in robo/hw1/starter.npz', 'rb'))
    cmap = data['arr_0'] #nxm =
    actions = data['arr_1'] #kx2
    observations = data['arr_2'] #k
    belief_states = data['arr_3'] #kxnxm

   
    Observations = np.array([2,0,0,2,1,0,1,1,1,2,1,1,1,1,1,2,2,0,0,1])
    Transition = np.array([[0.5,0.5],[0.5,0.5]])
    Emission = np.array([[0.4,0.1,0.5],[0.1,0.5,0.4]])
    id = np.array([0.5,0.5])

    hmm = HMM(Observations, Transition, Emission, id)
    
    alpha = hmm.forward()
    beta = hmm.backward()
    gamma = hmm.gamma_comp(alpha, beta)
    xi = hmm.xi_comp(alpha, beta, gamma)
    T_prime, M_prime, new_init_state = hmm.update(alpha, beta, gamma, xi)
    p_prime, p_original = hmm.trajectory_probability(alpha, beta, T_prime, M_prime, new_init_state)
    print("*****************************************************************")
    print("The value of alpha is:\n", alpha)
    print("*****************************************************************")
    print("The value of beta is:\n", beta)
    print("*****************************************************************")
    print("The value of gamma is:\n", gamma)
    print("*****************************************************************")
    print("The value of M_prime is:\n", M_prime)
    print("*****************************************************************")
    print("The value of T_prime is:\n", T_prime)
    print("*****************************************************************")
    print("The value of p_prime is:\n", p_prime)
    print("*****************************************************************")
    print("The value of p_original is:\n", p_original)
    print("*****************************************************************")
