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

# Q3 parameters:
q3_thetas = [0.05, 0.1, 0.5, 1, 2] # shape parameters
q3_lambdas = [0.001, 0.01, 0.1, 1] # regularization parameters
np.random.seed(42)

def rbf(dataset, validation=True, thetas=q3_thetas, lambdas=q3_lambdas):
    '''RBF model that minimizes least squares loss. Uses a Gaussian kernel with
    given list of shape parameter values (theta) and list of regularization 
    parameters (lambdas). Returns the best model (theta, lambda) and the
    corresponding RMSE validation loss. Construct the model using Cholesky
    factorization.
    
    The inputs to this function are:
    - dataset: string, either 'rosenbrock' or 'mauna_loa'
    - validation: boolean, if True, use the validation set to find the best. 
        If False, use the test set to find the best.
    - thetas: list of shape parameters
    - lambdas: list of regularization parameters

    The function iterates through different parameter combinations and prints 
    the RMSE loss for each combination. It then prints the minimum RMSE loss 
    found, which corresponds to the best parameter combination.
    '''

    min_rmse = np.inf

    # # Load the data:
    if dataset=='rosenbrock':   # shape of (1000,2)
        x_train, x_val, x_test, y_train, y_val, y_test = load_dataset('rosenbrock', n_train=1000, d=2)
        print("Dataset being tested: rosenbrock\n\n")
        x_train = x_train.reshape((-1, 2))
        x_val = x_val.reshape((-1, 2)) 

    elif dataset=='mauna_loa':  # shape of (709,1)
        x_train, x_val, x_test, y_train, y_val, y_test = load_dataset('mauna_loa')
        print("Dataset being tested: mauna_loa\n\n")
        x_train = x_train.reshape((-1, 1))
        x_val = x_val.reshape((-1, 1)) 

    y_train = y_train.reshape((-1, 1))
    y_val = y_val.reshape((-1, 1))

    if validation==True: 
        print("Using validation set to find best model\n\n")
        x = x_val
        y = y_val
    else:   # use x_test, y_test instead of x_val, y_val
        print("Using test set to find best model\n\n")
        x = x_test
        y = y_test

    for i, theta in enumerate(thetas):
        for j, lamb in enumerate(lambdas):
            # construct the kernel matrix
            K = gaussian_kernel(x_train, x_train, theta=theta)
            # add regularization to prevent overfitting
            K += lamb * np.eye(K.shape[0])
            # compute the cholesky factorization
            L = cho_factor(K)
            # compute the alpha values
            alpha = cho_solve(L, y_train.reshape(-1,1))
            # compute the RMSE loss
            # print("x_train shape = {}".format(x_train.shape))
            # print("x shape = {}".format(x.shape))
            # print("alpha shape = {}".format(alpha.shape))
            y_pred = np.dot(gaussian_kernel(x_train, x, theta=theta).T, alpha.reshape(-1,1)) # are these parameters right for gausian kernel? reshaping alpha into col vector
            rmse = np.sqrt(np.mean(np.square(y_pred - y))) # should this be least squared loss instead?? TODO
            min_rmse = min(min_rmse, rmse)
            print("theta = {}, lambda = {}, RMSE = {}\n".format(theta, lamb, rmse))
    print("min RMSE = {}".format(min_rmse))
    return

##############################################################################
#######################      QUESTION 4      #################################
##############################################################################

def greedy(dict, dataset='mauna_loa', validation=True):
    '''Greedy regression algorithm using a dictionary of basis functions. The
    inputs to this function are:
    - dict: dictionary of basis functions (contains at least 200)
    - dataset: string, either 'rosenbrock' or 'mauna_loa'
    - validation: boolean, if True, use the validation set to find the best.
        If False, use the test set to find the best.
    '''
    # Load the data:
    if dataset=='mauna_loa':
        x_train, x_val, x_test, y_train, y_val, y_test = load_dataset('mauna_loa')
        x_train = x_train.reshape((-1, 1))
        x_val = x_val.reshape((-1, 1)) 
        x_test = x_test.reshape((-1, 1)) #is this right?
    else:
        return None

    # Use either validation or test set:
    if validation==True:
        x = x_val
        y = y_val
    else:
        x = x_test
        y = y_test

##############################################################################
#######################         MAIN        ##################################
##############################################################################

if __name__ == "__main__":

    # Q3: RBF model
    print("Q3: RBF model")
    rbf('mauna_loa', True, q3_thetas, q3_lambdas) # RBF model on the validation set for mauna_loa
    # rbf('mauna_loa', False, q3_thetas, q3_lambdas) # RBF model on the test set for mauna_loa
    # rbf('rosenbrock', True, q3_thetas, q3_lambdas) # RBF model on the validation set for rosenbrock
    # rbf('rosenbrock', False, q3_thetas, q3_lambdas) # RBF model on the test set for rosenbrock

    # Q4: Greedy regression algorithm

    pass