import numpy as np
import pandas as pd
from collections import deque

data_df = pd.DataFrame(columns=['Outlook','Temperature','Humidity','Wind','PlayTennis'])

data_df.loc[0, 'Outlook'] = 'Sunny'
data_df.loc[1, 'Outlook'] = 'Sunny'
data_df.loc[2, 'Outlook'] = 'Overcast'
data_df.loc[3, 'Outlook'] = 'Rain'
data_df.loc[4, 'Outlook'] = 'Rain'
data_df.loc[5, 'Outlook'] = 'Rain'
data_df.loc[6, 'Outlook'] = 'Overcast'
data_df.loc[7, 'Outlook'] = 'Sunny'
data_df.loc[8, 'Outlook'] = 'Sunny'
data_df.loc[9, 'Outlook'] = 'Rain'
data_df.loc[10, 'Outlook'] = 'Sunny'
data_df.loc[11, 'Outlook'] = 'Overcast'
data_df.loc[12, 'Outlook'] = 'Overcast'
data_df.loc[13, 'Outlook'] = 'Rain'

data_df.loc[0, 'Temperature'] = 'Hot'
data_df.loc[1, 'Temperature'] = 'Hot'
data_df.loc[2, 'Temperature'] = 'Hot'
data_df.loc[3, 'Temperature'] = 'Mild'
data_df.loc[4, 'Temperature'] = 'Cool'
data_df.loc[5, 'Temperature'] = 'Cool'
data_df.loc[6, 'Temperature'] = 'Cool'
data_df.loc[7, 'Temperature'] = 'Mild'
data_df.loc[8, 'Temperature'] = 'Cool'
data_df.loc[9, 'Temperature'] = 'Mild'
data_df.loc[10, 'Temperature'] = 'Mild'
data_df.loc[11, 'Temperature'] = 'Mild'
data_df.loc[12, 'Temperature'] = 'Hot'
data_df.loc[13, 'Temperature'] = 'Mild'

data_df.loc[0, 'Humidity'] = 'High'
data_df.loc[1, 'Humidity'] = 'High'
data_df.loc[2, 'Humidity'] = 'High'
data_df.loc[3, 'Humidity'] = 'High'
data_df.loc[4, 'Humidity'] = 'Normal'
data_df.loc[5, 'Humidity'] = 'Normal'
data_df.loc[6, 'Humidity'] = 'Normal'
data_df.loc[7, 'Humidity'] = 'High'
data_df.loc[8, 'Humidity'] = 'Normal'
data_df.loc[9, 'Humidity'] = 'Normal'
data_df.loc[10, 'Humidity'] = 'Normal'
data_df.loc[11, 'Humidity'] = 'High'
data_df.loc[12, 'Humidity'] = 'Normal'
data_df.loc[13, 'Humidity'] = 'High'

data_df.loc[0, 'Wind'] = 'Weak'
data_df.loc[1, 'Wind'] = 'Strong'
data_df.loc[2, 'Wind'] = 'Weak'
data_df.loc[3, 'Wind'] = 'Weak'
data_df.loc[4, 'Wind'] = 'Weak'
data_df.loc[5, 'Wind'] = 'Strong'
data_df.loc[6, 'Wind'] = 'Strong'
data_df.loc[7, 'Wind'] = 'Weak'
data_df.loc[8, 'Wind'] = 'Weak'
data_df.loc[9, 'Wind'] = 'Weak'
data_df.loc[10, 'Wind'] = 'Strong'
data_df.loc[11, 'Wind'] = 'Strong'
data_df.loc[12, 'Wind'] = 'Weak'
data_df.loc[13, 'Wind'] = 'Strong'

data_df.loc[0, 'PlayTennis'] = 'No'
data_df.loc[1, 'PlayTennis'] = 'No'
data_df.loc[2, 'PlayTennis'] = 'Yes'
data_df.loc[3, 'PlayTennis'] = 'Yes'
data_df.loc[4, 'PlayTennis'] = 'Yes'
data_df.loc[5, 'PlayTennis'] = 'No'
data_df.loc[6, 'PlayTennis'] = 'Yes'
data_df.loc[7, 'PlayTennis'] = 'No'
data_df.loc[8, 'PlayTennis'] = 'Yes'
data_df.loc[9, 'PlayTennis'] = 'Yes'
data_df.loc[10, 'PlayTennis'] = 'Yes'
data_df.loc[11, 'PlayTennis'] = 'Yes'
data_df.loc[12, 'PlayTennis'] = 'Yes'
data_df.loc[13, 'PlayTennis'] = 'No'

data_df.head()

# separate target from predictors
X = np.array(data_df.drop('PlayTennis', axis=1).copy())
y = np.array(data_df['PlayTennis'].copy())
feature_names = list(data_df.keys())[:4]

from ID3 import DecisionTreeClassifier
tree_clf = DecisionTreeClassifier(X=X, attribute_names=feature_names, y=y)
print("System entropy {:.4f}".format(tree_clf.entropy))
tree_clf.id3()
tree_clf.printTree()

# # generate some data
# # define features and target values
# data = {
#     'wind_direction': ['N', 'S', 'E', 'W'],
#     'tide': ['Low', 'High'],
#     'swell_forecasting': ['small', 'medium', 'large'],
#     'good_waves': ['Yes', 'No']
# }

# # create an empty dataframe
# data_df = pd.DataFrame(columns=data.keys())

# np.random.seed(42)
# # randomnly create 1000 instances
# for i in range(1000):
#     data_df.loc[i, 'wind_direction'] = str(np.random.choice(data['wind_direction'], 1)[0])
#     data_df.loc[i, 'tide'] = str(np.random.choice(data['tide'], 1)[0])
#     data_df.loc[i, 'swell_forecasting'] = str(np.random.choice(data['swell_forecasting'], 1)[0])
#     data_df.loc[i, 'good_waves'] = str(np.random.choice(data['good_waves'], 1)[0])

# data_df.head()

# # separate target from predictors
# X = np.array(data_df.drop('good_waves', axis=1).copy())
# y = np.array(data_df['good_waves'].copy())
# feature_names = list(data_df.keys())[:3]


# from ID3 import DecisionTreeClassifier
# # instantiate DecisionTreeClassifier
# tree_clf = DecisionTreeClassifier(X=X, attribute_names=feature_names, y=y)
# print("System entropy {:.4f}".format(tree_clf.entropy))
# # run algorithm id3 to build a tree
# tree_clf.id3()
# tree_clf.printTree()