import numpy as np
import time
from data_utils import load_dataset
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve

##############################################################################
#######################      QUESTION 3      #################################
##############################################################################

# MULTIDIMENSIONAL KERNEL DEFINITION - from class notes
def gaussian_kernel(x, z, theta=1.):
        """
        Evaluate the Gram matrix for a Gaussian kernel between points in x and z.
        Inputs:
            x : array of shape (N, d)
            z : array of shape (M, d)
            theta : lengthscale parameter (>0)
        Outputs:
            k : Gram matrix of shape (N, M)
        """
        # reshape the matricies correctly for broadcasting
        x = np.expand_dims(x, axis=1)
        z = np.expand_dims(z, axis=0)

        # now evaluate the kernel using the euclidean distances squared between points
        return np.exp(-np.sum(np.square(x-z)/theta, axis=2, keepdims=False))

# Q2 parameters:
q2_thetas = [0.05, 0.1, 0.5, 1, 2] # shape parameters
q2_lambdas = [0.001, 0.01, 0.1, 1] # regularization parameters
np.random.seed(42)

def rbf(x_train, y_train, x_val, y_val, thetas=q2_thetas, lambdas=q2_lambdas):
    '''RBF model that minimizes least squares loss. Uses a Gaussian kernel with
    given list of shape parameter values (theta) and list of regularization 
    parameters (lambdas). Returns the best model (theta, lambda) and the
    corresponding RMSE validation loss. Construct the model using Cholesky
    factorization.'''

    # make inputs for  plotting like the "xx" in class notes? or is that already in given dataset

    for i, theta in enumerate(thetas):
        pass


    pass

##############################################################################
#######################      QUESTION 4      #################################
##############################################################################

##############################################################################
#######################         MAIN        ##################################
##############################################################################

if __name__ == "__main__":
    # # Load the data:
    # x_train, x_val, x_test, y_train, y_val, y_test = load_dataset('mauna_loa')
    # print("Dataset being tested: mauna_loa\n\n")
    # x_train, x_val, x_test, y_train, y_val, y_test = load_dataset('rosenbrock', n_train=5000, d=2)
    # print("Dataset being tested: rosenbrock\n\n")


    pass