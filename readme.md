# README

This contains information on how to run different parts of Question 3 and 
Question 4 from Assignment 2 (ROB313). In addition, the code is also throughly commented.
Ensure that you are cd'd into the correct folder (data), which contains the
necessary datasets to read into as well as the rob313-a2.py file. The code to 
run is in the rob313-a2.py file, and can be run in the terminal:

## Question 3

Question 3 constructs a radial basis function (RBF) model thaat minimizes the 
least-squares loss function using a Gaussian kernel. It is used to select
the optimal combination of shape and regularization parameters by evaluating 
on the validation set of mauna_loa and rosenbrock, and then finding the test RMSE 
by evaluating on the test set of mauna_loa and rosenbrock with this model. 

In order to use Question 3, there are different lines to be uncommented in the 
main function depending on what is desired. Each of the 4 lines are labelled
with comments describing what each one does: 
* RBF model on the validation set for mauna_loa
* RBF model on the test set for mauna_loa
* RBF model on the validation set for rosenbrock
* RBF model on the test set for rosenbrock

Each line will print into the terminal the different RMSE values resulting 
from either the test or validation set with its corresponding parameter 
combination of theta and lambda. 

## Question 4

Question 4 implements a greedy regression algorithm using a dictionary of 
over 400 basis functions for the mauna_loa dataset. The orthogonal matching 
pursuit metric is used to select a basis function at each iteration. The
stopping criterion is the minimum description length (MDL).

To run this function, simply uncomment the one line in the main function: 

        greedy()

The training and validation data are used to train the model. Following line 219 
of the greedy function, the code is used on the testing dataset of mauna_loa.
In the terminal, it prints the chosen basis functions and their selected
parameters, and it prints the resulting test RMSE.

Lines 252-258 are commented out and used for testing purposes to understand 
what the weights look like, but they can be uncommented if desired, and the 
results will also be printed in the terminal. After this code, there is the code
written to plot the y-predictions over the testing set. All of these functions
are carried out through uncommenting the greedy() line in the main function.