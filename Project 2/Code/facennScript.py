'''
Comparing single layer MLP with deep MLP (using TensorFlow)
'''
import numpy as np
import pickle

from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import time


# Do not change this
def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
                            
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W


def sigmoid(z):
    sigma = 1.0/(1.0 + np.exp(-z))
    return  sigma

def nnObjFunction(params, *args):
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
    
    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0
    
    # Your code here
    
    CodingSch = np.zeros((training_label.shape[0], 2))

    CodingSch[np.arange(training_label.shape[0], dtype="int"), training_label.astype(int)] = 1
    training_label = CodingSch
    
    #Forward Propagation
    training_data = np.column_stack((np.array(training_data), np.array(np.ones(training_data.shape[0]))))
    
    #Feed Forward
    #From input layer to hidden layers
    sigma1 = sigmoid(np.dot(training_data, w1.T)) #Dot Product of training data and transpose of w1
    sigma1Shape = np.ones(sigma1.shape[0])
    sigma1 = np.column_stack((sigma1, sigma1Shape))
    
    #From hidden layers to output layers
    sigma2 = sigmoid(np.dot(sigma1, w2.T))
    
    myDelta = sigma2 - training_label #Back propagation
    
    gradW2 = np.dot(myDelta.T, sigma1) #Calculate Gradient Descent
    gradW1 = np.delete(np.dot(((1 - sigma1) * sigma1 * (np.dot(myDelta, w2))).T, training_data), n_hidden, 0)
    
    N = training_data.shape[0]

    lamb_2n = (lambdaval / (2 * N))
    sumOfSquares1 = np.sum(np.square(w1))
    sumOfSquares2 = np.sum(np.square(w2))

    #Updating the obj_val 
    obj_val = ((np.sum(-1 * (training_label * np.log(sigma2) + (1 - training_label) * np.log(1 - sigma2)))) / N) + ( lamb_2n *( sumOfSquares1 + sumOfSquares2))
    
    gradW1 = (gradW1 + (lambdaval * w1)) / N
    gradW2 = (gradW2 + (lambdaval * w2)) / N
    
    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_grad = np.array([])
    obj_grad = np.concatenate((gradW1.flatten(), gradW2.flatten()), 0)
    
    return (obj_val, obj_grad)


def nnPredict(w1,w2,data):
    
    labels = np.array([])
    # Your code here
    lenData = np.ones( len(data)) #Adding bias of 1 to input vector

    data = np.column_stack( [data, lenData])
    
    sigma1 = np.column_stack( [sigmoid( data.dot( w1.T)), lenData]) #Sigmoid of data and transpose of w1
    sigma2 = sigmoid( sigma1.dot( w2.T)) #Sigmoid of result calculated in previous step and transpose of w2

    labels = np.argmax( sigma2, axis=1) #Max values for predicted labels
    
    return labels

# Do not change this
def preprocess():
    pickle_obj = pickle.load(file=open('face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255

    labels = labels[0]
    train_y = labels[0:21100]
    valid_y = labels[21100:23765]
    test_y = labels[23765:]
    return train_x, train_y, valid_x, valid_y, test_x, test_y

"""**************Neural Network Script Starts here********************************"""
startTime = time.time()
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
#  Train Neural Network
# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]
# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 64
# set the number of nodes in output unit
n_class = 2

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);
# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)
# set the regularization hyper-parameter
lambdaval = 5;
args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
opts = {'maxiter' :50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)
params = nn_params.get('x')
#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

#Test the computed parameters
predicted_label = nnPredict(w1,w2,train_data)
#find the accuracy on Training Dataset
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,validation_data)
#find the accuracy on Validation Dataset
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,test_data)
#find the accuracy on Validation Dataset
print('\n Test set Accuracy:' +  str(100*np.mean((predicted_label == test_label).astype(float))) + '%')

timeDifference = time.time()-startTime
print('\n Time Taken: ' + str(timeDifference)+ ' seconds.')
