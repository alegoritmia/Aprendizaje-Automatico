# https://machinelearningmastery.com/plot-a-decision-surface-for-machine-learning/
# decision surface for logistic regression on a binary classification dataset
from numpy import where
from numpy import meshgrid
from numpy import arange
from numpy import hstack
from matplotlib import pyplot
import numpy as np

def plot_decision_boundary(X, y, classifier):
    # for other color maps, see
    # https://matplotlib.org/stable/gallery/color/colormap_reference.html
    colormap = 'Greens'
    # unique classes:
    unique_classes = np.unique(y)
    # define bounds of the domain
    min1, max1 = X[:, 0].min()-1, X[:, 0].max()+1
    min2, max2 = X[:, 1].min()-1, X[:, 1].max()+1
    # define the x and y scale
    x1grid = arange(min1, max1, 0.1)
    x2grid = arange(min2, max2, 0.1)
    # create all of the columns and rows of the grid
    xx, yy = meshgrid(x1grid, x2grid)
    # flatten each grid to a vector
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
    # horizontal stack vectors to create x1,x2 input for the model
    grid = hstack((r1,r2))

    # TODO: Your classifier goes here
    # make predictions for the grid
    yhat = classifier.predict(grid)
    # reshape the predictions back into a grid
    zz = yhat.reshape(xx.shape)

    # plot the grid of x, y and z values as a surface
    pyplot.contourf(xx, yy, zz, cmap=colormap)

    # create scatter plot for samples from each class
    # for class_value in range(len(unique_classes)):
    for class_value in unique_classes:
        # get row indexes for samples with this class
        row_ix = where(y == class_value)[0] # need the first element of the tuple :S
        # create scatter of these samples
        pyplot.scatter(X[row_ix, 0], X[row_ix, 1], alpha=0.5, cmap=colormap)
    pyplot.show()

    # this is for classifiers with probability
    # import numpy as np
    # import pandas as pd
    # import matplotlib.pyplot as plt
    # from matplotlib.colors import ListedColormap
    # from sklearn import datasets
    # from sklearn.linear_model import LogisticRegression
    # from sklearn.preprocessing import StandardScaler
    # from sklearn.metrics import accuracy_score, confusion_matrix
    # from sklearn.model_selection import train_test_split

    # X, y = datasets.make_blobs(n_samples = 1000, centers = 2, n_features = 2, random_state = 1, cluster_std = 3)
    # # # create scatter plot for samples from each class
    # # for class_value in range(2):    # get row indexes for samples with this class
    # #     row_ix = np.where(y == class_value)    # create scatter of these samples
    # #     plt.scatter(X[row_ix, 0], X[row_ix, 1])# show the plot
    # # plt.show()


    # # regressor = LogisticRegression()# fit the regressor into X and y
    # # regressor.fit(X, y)# apply the predict method 
    # # y_pred = regressor.predict(X)

    # # Getting the min and max value for each feature (and giving an extra margin)
    # min1, max1 = X[:, 0].min() - 1, X[:, 0].max() + 1 #1st feature
    # min2, max2 = X[:, 1].min() - 1, X[:, 1].max() + 1 #2nd feature

    # # Then we define the coordinates
    # x1_scale = np.arange(min1, max1, 0.1)
    # x2_scale = np.arange(min2, max2, 0.1)

    # # Now we need to convert x1 and x2 into a grid
    # x_grid, y_grid = np.meshgrid(x1_scale, x2_scale)

    # # The generated x_grid is a 2-D array.
    # # To be able to use it, we need to reduce the size to a one dimensional array 
    # # using the flatten() method from the numpy library.

    # # flatten each grid to a vector
    # x_g, y_g = x_grid.flatten(), y_grid.flatten()
    # x_g, y_g = x_g.reshape((len(x_g), 1)), y_g.reshape((len(y_g), 1))

    # # Now, stacking the vectors side-by-side as columns in an input dataset, like the original dataset, but at a much higher resolution.
    # grid = np.hstack((x_g, y_g))

    # # define the model
    # model = LogisticRegression()
    # # fit the model
    # model.fit(X, y)
    # # make predictions for the grid
    # y_pred_2 = model.predict(grid)
    # #predict the probability
    # p_pred = model.predict_proba(grid)
    # # keep just the probabilities for class 0
    # p_pred = p_pred[:, 0]
    # # reshaping the results
    # pp_grid = p_pred.reshape(x_grid.shape)

    # # plot the grid of x, y and z values as a surface
    # surface = plt.contourf(x_grid, y_grid, pp_grid, cmap='Pastel1')
    # plt.colorbar(surface)# create scatter plot for samples from each class
    # for class_value in range(2):
    # # get row indexes for samples with this class
    #     row_ix = np.where(y == class_value)    # create scatter of these samples
    #     plt.scatter(X[row_ix, 0], X[row_ix, 1], cmap='Pastel1')# show the plot
    # plt.show()