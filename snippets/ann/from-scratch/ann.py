import numpy as np
import pandas as pd
from plot import plot_decision_boundary
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def z(input, Theta):
    # print('Theta is {} in transpose {}, input is {}'.format(Theta.shape, Theta.T.shape, input.shape))
    # we need to get an n x m array, where n is the neurons in this layer and m is the number of activations we used
    return Theta @ input

def activation(z):
    return sigmoid(z)

def initialize_weights(X, num_classes, hidden):
    weights = []
    out_neurons = num_classes 
    n = X.shape[0]
    
    h_layers = len(hidden)
    # print('Hidden layers will be {}'.format(h_layers))
    s_j = n
    # [2]
    for j in range(0,h_layers):
        s_jplus1 = hidden[j]
        cols = s_j + 1
        # print('Theta {} will be {}x{}'.format(j+1, s_jplus1, cols))
        # weights_layer = np.zeros((s_jplus1, cols))
        weights_layer = np.random.rand(s_jplus1, cols) # between 0,1
        # weights_layer = np.random.uniform(low=-10, high=10, size=(s_jplus1, cols)) # between a range
        weights.append(weights_layer)
        s_j = s_jplus1
    
    weights.append(np.random.rand(out_neurons, s_j + 1)) # between 0-1
    # weights.append(np.random.uniform(low=-10, high=10, size=(out_neurons, s_j + 1))) # between a range
    
    return weights

def print_matrix(matrix):
    for e in matrix:
        print('shape: {} {}'.format(e.shape, e))

def initialize_activations(X, output_neurons, hidden):
    # initialize activations, they are
    # input layer: (n+1) x m, 
    # hidden: (each + 1) x m,
    # output: output_neurons x m
    # So we can represent them with a list of length 2 + len(hidden)
    m = X.shape[1] # dimension 0 = features, dimension 1 = # examples
    activations = []
    
    # input layer
    a_1 = X
    biases = np.ones(m)
    a_1 = np.vstack((biases, a_1))
    activations.append(a_1)

    for i in range(len(hidden)):
        a_i = np.zeros((hidden[i] + 1 , m))
        a_i[0,:] = 1.0
        # a_i[0,0] = 1.0 # TODO: Initialize in 1 all activations for bias neurons. Previous line does it
        activations.append(a_i)

    # output layer
    activations.append(np.zeros((output_neurons, m)))
    return activations

def forward(X, hidden, activations, weights):
    # Hidden layers
    for i in range(0, len(hidden)):
        a_i = activations[i]
        # print('In Layer {}'.format(i+1))
        z_next = z(a_i, theta[i])
        # print('z_i are {}'.format(z_next.T))
        a_next = activation(z_next)
        # this line would fail for output layer, that's why process that separatedly 
        activations[i+1][1:] = a_next 
        # print('activations {} are {}'.format(i+1, a_next))

    # output layer (i+2 is the output layer)
    a_i = activations[i+1]
    z_next = z(a_i, theta[i+1])
    a_next = activation(z_next)
    activations[i+2] = a_next
    # print('output is {}'.format(a_next.T))

def backprop(y, hidden, activations, theta, alpha, reg):
    m = y.shape[1]
    delta = []
    # Calculating local gradients
    # Output layer
    y_pred = activations[-1]
    delta_i = y_pred - y
    # print('Error is {}'.format(delta_i.T))
    delta.append(delta_i)

    start = len(activations) - 1

    # Hidden layers
    for i in range(start, 1, -1): # we don't calculate errors for input layer
        # print('In layer {}'.format(i))
        theta_prev = theta[i-1][:,1:] # this is ignoring bias
        tmp = theta_prev.T @ delta_i
        # print('tmp is {}'.format(tmp.T))
        delta_i = tmp * ( activations[i-1][1:] * (1 - activations[i-1][1:])) # this is ignoring bias
        # print('Error is {}'.format(delta_i.T))
        delta.append(delta_i)
    
    # delta list holds the values
    # delta length is all but first layer (layers-1)
    # we could add an extra column for input layer to have the same
    # indexes as in the slides.
    delta.reverse()
    # print(delta)

    # Calculating Deltas
    num_clases = theta[-1].shape[0]
    Delta = initialize_weights(X, num_clases, hidden) # The Delta has the same dimensions as the weight matrix (we will ignore bias though)
    # This is for weights, not for neurons, that is why we reach layer 1
    start = len(delta) - 1
    for i in range(start, -1, -1):
        # print('Now in layer {}'.format(i+1))
        activations_this_layer = activations[i][1:,:] # this is ignoring bias
        delta_next_layer = delta[i]
        activations_times_delta = activations_this_layer@delta_next_layer.T
        Delta[i][:,1:] = activations_times_delta.T 
    
    D = [ x/m for x in Delta]

    for i in range(len(D)):
        d = D[i]
        t = theta[i]
        d[:,1:] += reg * t[:,1:]
    
    # update rule
    for i in range(len(theta)):
        t = theta[i]
        d = D[i]

        t[:, 1:] = t[:, 1:] - (alpha * d[:, 1:])

    # print(theta)

def get_config_for_example():
    X = np.array([[0.05], [0.10]])
    y = np.array([[0.01], [0.99]])

    hidden = [2]
    # el dos es por las neuronas en la capa de salida
    theta = initialize_weights(X, 2, hidden)

    theta_1 = theta[0]
    theta_1[0,0] = 0.35
    theta_1[0,1] = 0.15
    theta_1[0,2] = 0.20

    theta_1[1,0] = 0.35
    theta_1[1,1] = 0.25
    theta_1[1,2] = 0.30

    #####
    theta_2 = theta[1]
    theta_2[0,0] = 0.60
    theta_2[0,1] = 0.40
    theta_2[0,2] = 0.45

    theta_2[1,0] = 0.60
    theta_2[1,1] = 0.50
    theta_2[1,2] = 0.55

    return (X, y, hidden, theta)

def predict(X, theta):
    a_i = X
    # all theta layers
    for i in range(len(theta)):
        # print('In layer {}'.format(i+1))
        biases = np.ones(a_i.shape[1])
        a_i = np.vstack((biases, a_i))

        z_next = z(a_i, theta[i])
        a_i = activation(z_next)

    return a_i

def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.T

def cost_function(activations, y, theta, reg_factor):
    m = y.shape[1]
    y_pred = activations[-1]

    left_part = y * np.log(y_pred)
    right_part = (1 - y)*np.log(1 - y_pred)
    sum = np.sum(left_part + right_part)
    sum = -sum/m

    # Here, T[:, 1: ] means everything but the weight from the first row, 
    # that is the weight from bias
    reg_part_per_layer = [ np.sum(T[:, 1: ]**2) for T in theta]
    reg_part = np.sum(reg_part_per_layer) * reg_factor / (2*m)
    return sum + reg_part


if __name__ == "__main__":
    print('Starting...')
    data = pd.read_csv('blobs.csv')
    X = data.iloc[:,:-1].to_numpy().T   # all but last column of labels
    y = data.iloc[:,-1].to_numpy()      # the last col is class
    y = y.reshape(-1,1)  # to get an mx1 array and not (m,)
    unique_classes = len(np.unique(y))
    y = get_one_hot(y, unique_classes) # 1 -> [0, 1, 0], 2 -> [0, 0, 1], 0->[1, 0, 0]
    

    # # This part should go to a fit function >>
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X.T, y.T, test_size = 0.20)

    X_train, X_test, y_train, y_test = X_train.T, X_test.T, y_train.T, y_test.T
    # # For xor dataset
    # hidden = [6,6]
    # reg_factor = 0.0
    # alpha=0.5
    # epochs = 100000

    # # For blobs dataset
    hidden = [2,5]
    reg_factor = 0.0
    alpha=0.05
    epochs = 5000 # 50000

    # # For circles dataset
    # hidden = [9,9,9]
    # reg_factor = 0.0
    # alpha=0.05
    # epochs = 500000

    # # For moons dataset
    # hidden = [4,4,4]
    # reg_factor = 0.0
    # alpha=0.1
    # epochs = 500000


    theta = initialize_weights(X_train, unique_classes, hidden)
    activations = initialize_activations(X_train, unique_classes, hidden)

    costs = []
    
    for e in range(epochs):
         # print('Epoch {} '.format(e))
         forward(X_train, hidden, activations, theta)
         cost = cost_function(activations, y_train, theta, reg_factor)
         costs.append(cost)
         backprop(y_train, hidden, activations, theta, alpha, reg_factor)
    # << This part should go to a fit function
    
    plot1 = plt.figure(1)
    plt.plot(range(len(costs)), costs)

    plot_decision_boundary(X.T, y.T, predict, theta)
    
    # (X, y, hidden, theta) =  get_config_for_example()
    # activations = initialize_activations(X,output_neurons=2, hidden=hidden)
    # # print_matrix(theta)
    # # print_matrix(activations)
    # for i in range(10000):
    #     # print('Epoch {} '.format(i))
    #     forward(X, hidden, activations, theta)
    #     backprop(y, hidden, activations, theta, alpha=0.5, reg=0)
    # y_pred = predict(X, theta)
    # print('{} pred as {}, should be {}'.format(X.T, y_pred.T, y.T))
    