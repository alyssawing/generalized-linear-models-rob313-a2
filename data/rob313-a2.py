import numpy as np
import time
from data_utils import load_dataset
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve
import math

##############################################################################
#######################      QUESTION 3      #################################
##############################################################################

# MULTIDIMENSIONAL KERNEL DEFINITION - from class notes
def gaussian_kernel(x, z, theta=1.): 
        """
        Evaluate the Gram matrix for a Gaussian kernel between points in x and z.
        Inputs:
            x : array with shape (N, d)
            z : array with shape (M, d)
            theta : lengthscale parameter (>0)
        Outputs:
            k : Gram matrix with shape (N, M)
        """
        # reshape the matricies correctly for broadcasting
        x = np.expand_dims(x, axis=1)
        z = np.expand_dims(z, axis=0)

        # now evaluate the kernel using the euclidean distances squared between points
        return np.exp(-np.sum(np.square(x-z)/theta, axis=2, keepdims=False))

# Q3 parameters:
q3_thetas = [0.05, 0.1, 0.5, 1, 2] # given shape parameters
q3_lambdas = [0.001, 0.01, 0.1, 1] # given regularization parameters
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

    # Load the data:
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

    if validation==True: # to find the validation RMSEs
        print("Using validation set to find best model\n\n")
        x = x_val
        y = y_val
    else:   # use x_test, y_test instead of x_val, y_val to find test RMSEs
        print("Using test set to find best model\n\n")
        # print("x_train shape = {}".format(x_train.shape)) # (511,1)
        # print("x_val shape = {}".format(x_val.shape))   # (145,1)
        # print("y_train shape = {}".format(x_train.shape)) # (511,1)
        # np.concatenate((x_train, x_val), axis=0) # (656,1)
        # np.concatenate((y_train, y_val), axis=0)
        x = x_test
        y = y_test

    for i, theta in enumerate(thetas):
        for j, lamb in enumerate(lambdas):
            K = gaussian_kernel(x_train, x_train, theta=theta)  # kernel matrix
            K += lamb * np.eye(K.shape[0])  # regularization to prevent overfitting
            R = cho_factor(K)   # Cholesky factorization
            alpha = cho_solve(R, y_train.reshape(-1,1)) # estimated alphas
            # print("x_train shape = {}".format(x_train.shape))
            # print("x shape = {}".format(x.shape))
            # print("alpha shape = {}".format(alpha.shape))
            y_pred = np.dot(gaussian_kernel(x_train, x, theta=theta).T, alpha.reshape(-1,1)) # reshaping alpha into col vector (mxn) x (nx1) 
            rmse = np.sqrt(np.mean(np.square(y_pred - y))) # calculate the rmse loss 
            min_rmse = min(min_rmse, rmse)  # save the minimum loss so far
            print("theta = {}, lambda = {}, RMSE = {}\n".format(theta, lamb, rmse))
    print("overall min RMSE = {}".format(min_rmse))
    return

##############################################################################
#######################      QUESTION 4      #################################
##############################################################################

def sin(w, phi, x):
    '''Sine function with phase shift phit, angular frequency w, and input x.'''
    return np.sin(w*x + phi)

def exponential(a, x):
    '''Exponential function with amplitude a, decay rate b, constant shift c,
    and input x.'''
    return np.exp(a*x) 

def polynomial(degree, x):
    '''Polynomial function with given degree and input x.'''
    return np.power(x, degree) 

def basis_maker():
    '''Returns a list of basis functions. The basis_maker creates ____ candidate
    basis functions, and the format of each element is a tuple, where the first
    value is the basis function and the second is the parameters to use.'''
    functions = []

    # Iterate over possible parameters for the sinusoidal function: 
    for w in np.arange(50, 150, 5): # 20 possibilities
        for phi in np.arange(-np.pi, np.pi, 0.3): # 21 possibilities
            functions.append((sin, {'w': w, 'phi': phi}))

    # Iterate over possible parameters for the exponential function:
    for a in np.arange(0.1, 1, 0.5):
        functions.append((exponential, {'a': a}))

    # Iterate over possible parameters for the polynomial function:
    for degree in range(1,6): # 5 possibilities
        functions.append((polynomial, {'degree': degree, 'c': 0}))
        
    return functions

def greedy(bases=basis_maker(), dataset='mauna_loa'):
    '''Greedy regression algorithm using a dictionary of basis functions. The
    inputs to this function are:
    - bases: dictionary of basis functions (contains at least 200)
    - dataset: string, either 'rosenbrock' or 'mauna_loa'

    At each iteration, use the orthogonal matching pursuit metric. The stopping
    criterion is the minimum description length (MDL) given in Q4. The function
    can be used to plot the prediction relative to the test data and the RMSE. 
    '''
    # Load the data:
    if dataset=='mauna_loa':
        x_train, x_val, x_test, y_train, y_val, y_test = load_dataset('mauna_loa')
        x_train = x_train.reshape((-1, 1))
        x_val = x_val.reshape((-1, 1)) 
        x_test = x_test.reshape((-1, 1)) #is this right?
    else:
        return None

    # To test, concatenate the training and validation sets:
    x_train = np.concatenate((x_train, x_val), axis=0)
    y_train = np.concatenate((y_train, y_val), axis=0)
    
    # Calculate the stopping criterion: 
    phi = [] # initialize the phi matrix of chosen basis functions (evaluated at x_train points)
    chosen = [] # list of chosen basis functions
    k = len(chosen) # number of functions picked so far
    N = x_train.shape[0]  # number of data points

    r = y_train   # intialize r (the residual)
    lste = np.linalg.norm(r)  # least-squares training error
    mdl = N/2*np.log(lste) + k/2*np.log(N) # intialize minimum description length as stop criterion 
    new_mdl = mdl

    while np.linalg.norm(r) > 0.1 and new_mdl <= mdl: # ensuring the error doesn't grow
        k += 1
        best_reduction = -float('inf') # initialize error to -infinity 

        # Pick a new basis function from the dictionary:
        for i in range(len(bases)):
            tup = bases[i]
            if tup[0] == sin:
                phi_col = tup[0](tup[1]['w'], tup[1]['phi'], x_train) # evaluate the basis function at x_train points
            elif tup[0] == exponential:
                phi_col = tup[0](tup[1]['a'], x_train)
            elif tup[0] == polynomial:
                phi_col = tup[0](tup[1]['degree'], x_train)
            # col = col.reshape((-1, 1)) # reshape into column vector
            J = np.square(np.dot(phi_col.T, r))/np.dot(phi_col.T, phi_col) # calculate J
            
            if not math.isnan(J):   # in case division by 0 
                if J > best_reduction:
                    best_reduction = J
                    best_basis = bases[i] # tuple format (basis function, parameters)
                    best_col = phi_col # column vector of basis function evaluated at x_train points

        chosen.append(best_basis) # add the best basis function to the list of chosen functions
        bases.remove(best_basis)    # remove best basis from bases

        # add the new column to phi matrix:
        if k == 1: # first iteration; phi is an empty list 
            phi = best_col
        else:
            phi = np.hstack((phi, best_col)) # add column to phi matrix

        # solve for the weights:
        phi_pinv = np.linalg.pinv(phi) # find pseudo-inverse of phi matrix
        w = np.dot(phi_pinv, y_train) # calculate the weights
        
        # update residual r and mdl:
        r = y_train - np.dot(phi, w)
        mdl = new_mdl
        new_mdl = N/2*np.log(np.linalg.norm(r)) + k/2*np.log(N) 

        # plot the prediction relative to the test data for each iteration for the TRAINING + VALIDATION SET:
        # this can be used to see how accuracy improves with each iteration
        # plt.plot(x_train, y_train, 'og', label='training data', markersize=2)
        # plt.plot(x_train, np.dot(phi, w), 'om', label='prediction', markersize=2)
        # plt.show()

    # FOR THE TESTING SET:
    final_phi = []
    count = 0

    for f in chosen:
        if f[0] == sin:
            if count == 0:
                final_phi = f[0](f[1]['w'], f[1]['phi'], x_test)
                count += 1
            else:
                final_phi = np.hstack((final_phi, f[0](f[1]['w'], f[1]['phi'], x_test)))
        elif f[0] == exponential:
            if count == 0:
                final_phi = f[0](f[1]['a'], x_test)
                count += 1
            else:
                final_phi = np.hstack((final_phi, f[0](f[1]['a'], x_test)))
        elif f[0] == polynomial:
            if count == 0:
                final_phi = f[0](f[1]['degree'], x_test)
                count += 1
            else:
                final_phi = np.hstack((final_phi, f[0](f[1]['degree'], x_test)))      

    # Get the y predictions:
    y_pred = np.dot(final_phi, w)

    # Get the test RMSE
    rmse = np.sqrt(np.mean(np.square(y_test - y_pred)))
    print("testing RMSE: ", rmse)

    # Plot the TESTING PREDICTIONS:
    # plt.plot(x_train, y_train, 'ob', label='training data', markersize=2)
    plt.plot(x_test, y_test, 'og', label='test data', markersize=5)
    plt.plot(x_test, y_pred, 'or', label='prediction', markersize=5)
    plt.title('Greedy Regression Algorithm on ' + dataset + ' dataset')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()
    
    return chosen, w

##############################################################################
#######################         MAIN        ##################################
##############################################################################

if __name__ == "__main__":

    ########################### Q3: RBF model ##################################
    print("Q3: RBF model")
    # rbf('mauna_loa', True, q3_thetas, q3_lambdas) # RBF model on the validation set for mauna_loa
    # rbf('mauna_loa', False, q3_thetas, q3_lambdas) # RBF model on the test set for mauna_loa
    # rbf('rosenbrock', True, q3_thetas, q3_lambdas) # RBF model on the validation set for rosenbrock
    # rbf('rosenbrock', False, q3_thetas, q3_lambdas) # RBF model on the test set for rosenbrock

    #################### Q4: Greedy regression algorithm #######################
    print("Q4: Greedy regression algorithm")
    greedy() # greedy regression algorithm on the mauna_loa dataset

    # Extra plotting to help determine parameters to try for mauna_loa in designing basis function list:
    # Plotting mauna_loa dataset (training points):
    # x_train, x_val, x_test, y_train, y_val, y_test = load_dataset('mauna_loa')
    # plt.scatter(x_train, y_train, color='blue',label='training points', s=1)
    # x = np.arange(min(x_train), max(x_train), 0.01)
    # y = 0.1*np.sin(80*x)+x
    # plt.plot(x, y, color='red', markersize=2,label='0.1*sin(80*x)+x')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.legend()
    # plt.show()