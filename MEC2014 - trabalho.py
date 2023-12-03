# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 10:15:22 2023

@author: pepereira
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import warnings
warnings.filterwarnings('once') 
warnings.filterwarnings("ignore", category=DeprecationWarning)
from sklearn.metrics import balanced_accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV , RandomizedSearchCV

input_data = pd.read_excel('C:/Users/pepereira/Downloads/reference_data_transf_gas.xlsx')
del input_data['Unnamed: 0']

input_data['fault'].value_counts()
dict_fault = {0:'PD',1:'D1',2:'D2',3:'T1',4:'T2',5:'T3'}



input_data['R1'] = (input_data['CH4']+input_data['C2H6']) / (input_data['H2']+input_data['CH4']+input_data['C2H6']+input_data['C2H4']+input_data['C2H2'])
input_data['R2'] = (input_data['CH4']+input_data['C2H4']) / (input_data['H2']+input_data['CH4']+input_data['C2H6']+input_data['C2H4']+input_data['C2H2'])
input_data['R3'] = input_data['C2H6'] / (input_data['CH4'] + input_data['C2H4'])
input_data['R4'] = (input_data['CH4']+input_data['H2']) / (input_data['H2']+input_data['CH4']+input_data['C2H6']+input_data['C2H4']+input_data['C2H2'])
input_data['R5'] = (input_data['C2H4']+input_data['C2H2']) / (input_data['H2']+input_data['CH4']+input_data['C2H6']+input_data['C2H4']+input_data['C2H2'])
input_data['R6'] = input_data['C2H2'] / input_data['C2H4']
input_data['R6'] = input_data['R6'].replace(np.nan,0)



cols = ['H2', 'CH4', 'C2H2', 'C2H4', 'C2H6', 'CO', 'CO2',
        'R1', 'R2', 'R3', 'R4', 'R5', 'R6']





from collections import Counter
counter = Counter(input_data['fault'])
print('\nOriginal:')
for k,v in counter.items():
    per = v / len(input_data['fault']) * 100
    print('Class=%d, n=%d (%.3f%%)' % (k, v, per))



# EDA
import seaborn as sns
input_data.shape
input_data.describe()
input_data['fault'].unique()
input_data['fault'].value_counts()

sns.heatmap(input_data[cols].corr())

input_data[cols].hist(figsize=(20, 10))
plt.tight_layout()
plt.show()

transformed_df = input_data.copy()
from sklearn.preprocessing import StandardScaler
scaled_features = StandardScaler().fit_transform(input_data[cols])
transformed_df[cols] = scaled_features


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(transformed_df[cols], input_data['fault'], test_size=0.3,stratify=input_data['fault'])


print('\nTrain:')
counter=Counter(y_train)
for k,v in counter.items():
    per = v / len(y_train) * 100
    print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
print('\nTest:')
counter=Counter(y_test)
for k,v in counter.items():
    per = v / len(y_test) * 100
    print('Class=%d, n=%d (%.3f%%)' % (k, v, per))


from sklearn.model_selection import RepeatedStratifiedKFold
cvFold = RepeatedStratifiedKFold(n_splits=5, n_repeats=5)


# --------------------------------- Linear model (Ridge) ---------------------------------
print('\n--------------------------------- (Linear model (Ridge)) ---------------------------------')

from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import classification_report, confusion_matrix 

# Create parameters for grid search
ridge_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}

# Perform grid search for RidgeClassifier
grid_ridge = GridSearchCV(RidgeClassifier(), ridge_grid, refit=True, verbose=3,
                           cv=cvFold, scoring='balanced_accuracy')

# Fit the model for grid search
grid_ridge.fit(X_train, y_train)

# Print the best parameters after tuning
print(grid_ridge.best_params_)

# Print how the model looks after hyper-parameter tuning
print(grid_ridge.best_estimator_)

# Make predictions using the RidgeClassifier with grid search
grid_predictions_ridge = grid_ridge.predict(X_test)

# Print metrics for the RidgeClassifier with grid search
print("Ridge Classifier (test):")
print("Balanced Accuracy:", balanced_accuracy_score(y_test, grid_predictions_ridge))
print("f1 weighted:", metrics.f1_score(y_test, grid_predictions_ridge, average='weighted'))
print("precision weighted:", metrics.precision_score(y_test, grid_predictions_ridge, average='weighted'))
print("recall weighted:", metrics.recall_score(y_test, grid_predictions_ridge, average='weighted'))
print(classification_report(y_test, grid_predictions_ridge))
print(np.round(confusion_matrix(y_test, grid_predictions_ridge, normalize='true'), 2))



# --------------------------------- SVM ---------------------------------
print('\n--------------------------------- (SVM) ---------------------------------')

from sklearn import svm
clf_svm = svm.SVC(C=0.1, gamma=1, kernel='linear')
clf_svm.fit(X_train, y_train)
y_pred = clf_svm.predict(X_test)

from sklearn import metrics
print("Balanced Accuracy:",balanced_accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))





svm_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['linear']}

grid_svm = GridSearchCV(svm.SVC(), svm_grid, refit = True, verbose = 3,
                        cv=cvFold,scoring='balanced_accuracy')
# fitting the model for grid search 
grid_svm.fit(X_train, y_train)
# print best parameter after tuning 
print(grid_svm.best_params_)
# print how our model looks after hyper-parameter tuning 
print(grid_svm.best_estimator_)


grid_predictions_svm = grid_svm.predict(X_test)


print("SVM (teste):")
print("Balanced Accuracy:",balanced_accuracy_score(y_test, grid_predictions_svm))
#print("f1 weighted:",metrics.f1_score(y_test, grid_predictions_svm,average='weighted'))
print("f1 weighted:",metrics.f1_score(y_test, grid_predictions_svm,average='weighted'))
print("precision weighted:",metrics.precision_score(y_test, grid_predictions_svm,average='weighted'))
print("recall weighted:",metrics.recall_score(y_test, grid_predictions_svm,average='weighted'))
print(classification_report(y_test, grid_predictions_svm))
print(np.round(confusion_matrix(y_test, grid_predictions_svm,normalize='true'),2))


from sklearn.decomposition import PCA
pca = PCA(n_components=5)
pca_fit = pca.fit(transformed_df.loc[:,cols])
components  = pca_fit.transform(transformed_df.loc[:,cols])

plt.scatter(x=components[:,0],y=components[:,1],c=transformed_df['fault']) # px.scatter(components, x=0, y=1, color=input_data['fault'])
plt.show()





# --------------------------------- kNN ---------------------------------
print('\n--------------------------------- (kNN) ---------------------------------')
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV

knn = KNeighborsClassifier()
n_neighbors = [3,4,5,6,7,8,9,10]
weights = ['uniform' , 'distance']
algorithm = ['auto' , 'ball_tree' , 'kd_tree' ,'brute']
knn_grid = dict(n_neighbors = n_neighbors , weights = weights ,algorithm = algorithm )


gridSearch = GridSearchCV(knn, knn_grid, refit = True, verbose = 3,
                          cv=cvFold, n_jobs=-1, scoring='balanced_accuracy')

searchResults = gridSearch.fit(X_train.values, y_train.values)

best_knn = searchResults.best_estimator_
print(gridSearch.best_params_)
print(gridSearch.best_estimator_)

grid_predictions_knn = best_knn.predict(X_test)

print('kNN:')
print("Balanced Accuracy:",balanced_accuracy_score(y_test, grid_predictions_knn))
print("f1 weighted:",metrics.f1_score(y_test, grid_predictions_knn,average='weighted'))
print("precision weighted:",metrics.precision_score(y_test, grid_predictions_knn,average='weighted'))
print("recall weighted:",metrics.recall_score(y_test, grid_predictions_knn,average='weighted'))
print(classification_report(y_test, grid_predictions_knn))
print(np.round(confusion_matrix(y_test, grid_predictions_knn,normalize='true'),2))


# --------------------------------- Random Forest ---------------------------------
print('\n--------------------------------- (Random Forest) ---------------------------------')
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
n_estimators = [200, 1000, 2000] # [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['sqrt']
max_depth = [10, 50, 100] # [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

rf_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


rfSearch = GridSearchCV(rf, rf_grid, refit = True, verbose = 0, cv=cvFold, n_jobs=-1)
rfSearch_results = rfSearch.fit(X_train.values, y_train.values)

best_rf = rfSearch_results.best_estimator_
print(rfSearch.best_params_)
print(rfSearch.best_estimator_)

grid_predictions_rf = best_rf.predict(X_test)

print('Random Forest:')
print("Balanced Accuracy:",balanced_accuracy_score(y_test, grid_predictions_rf))
print("f1 weighted:",metrics.f1_score(y_test, grid_predictions_rf,average='weighted'))
print("precision weighted:",metrics.precision_score(y_test, grid_predictions_rf,average='weighted'))
print("recall weighted:",metrics.recall_score(y_test, grid_predictions_rf,average='weighted'))
print(classification_report(y_test, grid_predictions_rf))
print(np.round(confusion_matrix(y_test, grid_predictions_rf,normalize='true'),2))





# --------------------------------- NN ---------------------------------
print('\n--------------------------------- (NN) ---------------------------------')
import keras
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
#from scikeras.wrappers import KerasClassifier
#from keras.utils import np_utils --> DEPRECATED
from keras import utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam
import tensorflow as tf


# encode class values as integers
encoder = LabelEncoder()
encoder.fit(input_data['fault'])
encoded_Y = encoder.transform(input_data['fault'])
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = utils.to_categorical(encoded_Y)


# build a model
model = Sequential()
# model.add(Dense(16, input_shape=(transformed_df[cols].shape[1],), activation='relu')) # input shape is (features,)
# model.add(Dense(8, activation='relu'))
# model.add(Dense(5, activation='softmax'))

model.add(Dense(32, input_shape=(transformed_df[cols].shape[1],)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(5, activation='softmax'))
model.summary()



# Define a custom metric function for balanced accuracy

def balanced_accuracy(y_true, y_pred):
    y_true_class = tf.argmax(y_true, axis=1)
    y_pred_class = tf.argmax(y_pred, axis=1)
    
    # Calculate balanced accuracy using TensorFlow functions
    true_positive = tf.cast(tf.math.count_nonzero(y_true_class * y_pred_class), dtype=tf.float32)
    true_negative = tf.cast(tf.math.count_nonzero((1 - y_true_class) * (1 - y_pred_class)), dtype=tf.float32)
    false_positive = tf.cast(tf.math.count_nonzero((1 - y_true_class) * y_pred_class), dtype=tf.float32)
    false_negative = tf.cast(tf.math.count_nonzero(y_true_class * (1 - y_pred_class)), dtype=tf.float32)
    
    sensitivity = true_positive / (true_positive + false_negative + tf.keras.backend.epsilon())
    specificity = true_negative / (true_negative + false_positive + tf.keras.backend.epsilon())
    
    balanced_acc = (sensitivity + specificity) / 2.0
    
    return balanced_acc

# compile the model
model.compile(optimizer=Adam(learning_rate=0.001), 
              loss='categorical_crossentropy',
              metrics=[balanced_accuracy])


# early stopping callback
# This callback will stop the training when there is no improvement in  
# the validation loss for 10 consecutive epochs.  
es = EarlyStopping(monitor='val_loss', 
                    mode='min',
                    patience=100, 
                    restore_best_weights=True) # important - otherwise you just return the last weigths...

X_train_NN, X_test_NN, y_train_NN, y_test_NN = train_test_split(transformed_df[cols], dummy_y, test_size=0.3)

# now we just update our model fit call
history = model.fit(X_train_NN,
                    y_train_NN,
                    callbacks=[es],
                    epochs=8000000, # you can set this to a big number!
                    batch_size=10,
                    shuffle=True,
                    validation_split=0.2,
                    verbose=0)


history_dict = history.history

# learning curve
# accuracy
acc = history_dict['balanced_accuracy']
val_acc = history_dict['val_balanced_accuracy']

# loss
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)



from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

preds = model.predict(X_train_NN) # see how the model did!
print(preds[0]) # i'm spreading that prediction across three nodes and they sum to 1
print(np.sum(preds[0])) # sum it up! Should be 1

matrix = confusion_matrix(y_train, preds.argmax(axis=1))
print(matrix)

print(classification_report(y_train, preds.argmax(axis=1)))


preds_test = model.predict(X_test_NN)
print('NN:')
print("Balanced Accuracy:",balanced_accuracy_score(y_test, preds_test.argmax(axis=1)))
print("f1 weighted:",metrics.f1_score(y_test, preds_test.argmax(axis=1),average='weighted'))
print("precision weighted:",metrics.precision_score(y_test, preds_test.argmax(axis=1),average='weighted'))
print("recall weighted:",metrics.recall_score(y_test, preds_test.argmax(axis=1),average='weighted'))
print(classification_report(y_test, preds_test.argmax(axis=1)))
print(np.round(confusion_matrix(y_test, preds_test.argmax(axis=1),normalize='true'),2))



# --------------------------------- SOM ---------------------------------
print('\n--------------------------------- (SOM) ---------------------------------')



from minisom import MiniSom
som_grid_rows = 1
som_grid_columns = 6
iterations = 20000
sigma = 1
learning_rate = 0.5

# define SOM:
som = MiniSom(x = som_grid_rows, y = som_grid_columns, input_len=13, sigma=sigma, learning_rate=learning_rate)

# Training
som.train_batch(X_train_NN.values, iterations, verbose=True)

# each neuron represents a cluster
winner_coordinates = np.array([som.winner(x) for x in X_train_NN.values]).T
# with np.ravel_multi_index we convert the bidimensional
# coordinates to a monodimensional index
cluster_index = np.ravel_multi_index(winner_coordinates, (som_grid_rows,som_grid_columns))


# plotting the clusters using the first 2 dimentions of the data
for c in np.unique(cluster_index):
    plt.scatter(X_train_NN.values[cluster_index == c, 0],
                X_train_NN.values[cluster_index == c, 1], label='cluster='+str(c), alpha=.7)

# plotting centroids
for centroid in som.get_weights():
    plt.scatter(centroid[:, 0], centroid[:, 1], marker='x', 
                s=8, linewidths=3, color='k', label='centroid')
plt.legend()




# classification
def classify(som, data):
    """Classifies each sample in data in one of the classes definited
    using the method labels_map.
    Returns a list of the same length of data where the i-th element
    is the class assigned to data[i].
    """
    winmap = som.labels_map(X_train_SOM.values, y_train_SOM)
    default_class = np.sum(list(winmap.values())).most_common()[0][0]
    result = []
    for d in data:
        win_position = som.winner(d)
        if win_position in winmap:
            result.append(winmap[win_position].most_common()[0][0])
        else:
            result.append(default_class)
    return result



X_train_SOM, X_test_SOM, y_train_SOM, y_test_SOM = train_test_split(transformed_df[cols], transformed_df['fault'], test_size=0.3, stratify=transformed_df['fault'])
som = MiniSom(13, 13, 13, sigma=1, learning_rate=0.05, 
              neighborhood_function='triangle', random_seed=10)
som.pca_weights_init(X_train_SOM.values)
som.train_random(X_train_SOM.values, 10000, verbose=0)


print('SOM:')
print("Balanced Accuracy:",balanced_accuracy_score(y_test_SOM, classify(som, X_test_SOM.values)))
print("f1 weighted:",metrics.f1_score(y_test_SOM, classify(som, X_test_SOM.values),average='weighted'))
print("precision weighted:",metrics.precision_score(y_test_SOM, classify(som, X_test_SOM.values),average='weighted'))
print("recall weighted:",metrics.recall_score(y_test_SOM, classify(som, X_test_SOM.values),average='weighted'))
print(classification_report(y_test_SOM, classify(som, X_test_SOM.values)))
print(np.round(confusion_matrix(y_test_SOM, classify(som, X_test_SOM.values),normalize='true'),2))






