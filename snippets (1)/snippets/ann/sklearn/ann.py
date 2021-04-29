import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from plot import plot_decision_boundary
from matplotlib import pyplot as plt

# Uncomment as needed
# # for the xor dataset
# df = pd.read_csv('xor.csv')
# X = df[['x1', 'x2']]
# y = df[['y']]

# # 6 with alpha 0.05
# mlp = MLPClassifier(hidden_layer_sizes=(6,6), max_iter=2000000, alpha=0.5) # alpha was 0.05
# mlp.fit(X, y.values.ravel()) # too few examples, so we use all of them (4)
# predictions = mlp.predict(X)
# print(predictions)
# plot_decision_boundary(X.to_numpy(), y.to_numpy(), mlp)
# plt.show()

# # for the synthetic datasets
# # df = pd.read_csv('blobs.csv')
# # df = pd.read_csv('circles.csv')
df = pd.read_csv('moons.csv')

X = df[['x1', 'x2']]
y = df[['y']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

# # for the blobs dataset
# # mlp = MLPClassifier(hidden_layer_sizes=(4,4,2), max_iter=1000000, alpha=0.05)

# # for the circles dataset
# # start with (4,4,2) and see the mess... (9,9,9) is working ok
# # mlp = MLPClassifier(hidden_layer_sizes=(9,9,9), max_iter=1000000, alpha=0.01)

# for the moons dataset: (3,3,3,3) starts to go well, (4,4,4) too
mlp = MLPClassifier(hidden_layer_sizes=(4,4,4), max_iter=1000000, alpha=0.1) # the weight 0.05 is causing some issues

mlp.fit(X_train, y_train.values.ravel())

predictions = mlp.predict(X_test)
print(predictions)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

plot_decision_boundary(X.to_numpy(), y.to_numpy(), mlp)
plt.show()