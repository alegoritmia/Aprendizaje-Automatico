from plot import plot_decision_boundary
from LogisticRegressor import LogisticRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":
    data = pd.read_csv('dataset-1.csv')
    X = data.iloc[:,:-1].to_numpy().T # all but last column of labels
    the_ones = np.ones((X.shape[1],1)).T # (1xm) additional row of 1's
    X = np.row_stack((the_ones, X)) # adding the row of 1's
    y = data.iloc[:,-1].to_numpy() # the last col is class
    y = y.reshape(y.shape[0],-1).T # this is to get (1,m) rather than an (m,) array (2d instead of 1d)
    
    # 0.001
    # 100000  38.73906394658962
    # 500000  25.217014015201844
    # 1000000 22.46541908824744
    # 0.002
    # 1000000 20.96603586007431
    # 1500000 20.612243806979556
    # 0.003
    # 1000000 20.49306802307075
    lr = LogisticRegressor(alpha=0.001, epochs=100000)
    lr.fit(X,y)

    costs = lr.costs
    plot1 = plt.figure(1)
    plt.plot(range(len(costs)), costs)
    # plt.show()

    idx = 1
    X_test = X[:,idx]
    y_pred = lr.predict(X_test)
    print('{} predicted as {} (was {})'.format(X_test, y_pred, y[:,idx]))


    class_accepted = data.loc[data['accepted'] == 1]
    class_not_accepted = data.loc[data['accepted'] == 0]
    plot2 = plt.figure(2)
    plt.scatter(class_accepted['mark1'], class_accepted['mark2'])
    plt.scatter(class_not_accepted['mark1'], class_not_accepted['mark2'])
    plt.title("Plot"); plt.xlabel('x1'); plt.ylabel('x2') 
    
    plot_decision_boundary(X.T, y.T, lr)
    plt.show()

    # For dataset 2
    # data = pd.read_csv('dataset-2-modified.csv')
    # X = data.iloc[:,:-1].to_numpy().T # all but last column of labels
    # y = data.iloc[:,-1].to_numpy() # the last col is class
    # y = y.reshape(y.shape[0],-1).T # this is to get (1,m) rather than an (m,) array (2d instead of 1d)
    
    # # 0.001
    # # cost 21.5011265699403 con 1500000 iters NO REG
    # # cost 21.041325254347985 con 2000000 iters
    # lr = LogisticRegressor(alpha=0.001, epochs=2000000, regularize=True)
    # lr.fit(X,y)

    # costs = lr.costs
    # plot1 = plt.figure()
    # plt.plot(range(len(costs)), costs)
    # plt.show()

    # idx = 1
    # X_test = X[:,idx]
    # y_pred = lr.predict(X_test)
    # print('{} predicted as {} (was {})'.format(X_test, y_pred, y[:,idx]))