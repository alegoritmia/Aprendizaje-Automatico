# https://www.datacamp.com/community/tutorials/decision-tree-classification-python
# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

# Visualizing the tree
from six import StringIO
from sklearn.tree import export_graphviz # to convert to a dot file
import pydotplus # to convert the dot file to an image
from IPython.display import Image 

data_df = pd.DataFrame(columns=['Outlook','Temperature','Humidity','Wind','PlayTennis'])
# data_df.loc[len(data_df.index)] = [list of values] 
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

mapping_dict_outlook = {'Sunny':0, 'Overcast':1, 'Rain':2}
mapping_dict_temperature = {'Hot':0, 'Cool':1, 'Mild':2}
mapping_dict_humidity = {'Normal':0, 'High':1}
mapping_dict_wind = {'Weak':0, 'Strong':1}
mapping_dict_playtennis = {'No':0, 'Yes':1}


feature_cols = ['Outlook','Temperature','Humidity','Wind']

# Convert the categories to a number. Currently sklearn does not support categories
encoding_outlook = pd.factorize(data_df['Outlook'])
data_df['Outlook'] = encoding_outlook[0]

encoding_temperature = pd.factorize(data_df['Temperature'])
data_df['Temperature'] = encoding_temperature[0]

encoding_humidity = pd.factorize(data_df['Humidity'])
data_df['Humidity'] = encoding_humidity[0]

encoding_wind = pd.factorize(data_df['Wind'])
data_df['Wind'] = encoding_wind[0]

encoding_play = pd.factorize(data_df['PlayTennis'])
data_df['PlayTennis'] = encoding_play[0]
print('Outlook encoding is {}'.format(encoding_outlook[0]))
print('Temperature encoding is {}'.format(encoding_temperature[0]))
print('Humidity encoding is {}'.format(encoding_humidity[0]))
print('Wind encoding is {}'.format(encoding_wind[0]))
print('Play encoding is {}'.format(encoding_play[0]))

X = data_df[feature_cols]
y = data_df.PlayTennis

# we have just a few examples, we just use all of them
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# clf = DecisionTreeClassifier()
clf = DecisionTreeClassifier(criterion="entropy")
clf = clf.fit(X, y)

to_classify = X.iloc[0] # day 1
y_pred = clf.predict([to_classify])
print('{} is classified as {}'.format(to_classify, y_pred))

to_classify = X.iloc[4] # day 5
y_pred = clf.predict([to_classify])
print('{} is classified as {}'.format(to_classify, y_pred))


dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('result.png')
Image(graph.create_png())
