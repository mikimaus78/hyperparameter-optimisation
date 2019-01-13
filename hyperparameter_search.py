
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
import time

import random

seed = 7

random.seed(seed)
"""

from numpy import loadtxt
from urllib.request import urlopen
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
raw_data = urlopen(url)
dataset = loadtxt(raw_data, delimiter=",")
print(dataset.shape)
"""

# We placed the dataset under datasets/ sub folder
DATASET_PATH = 'dataset/'

# We read the data from the CSV file
data_path = os.path.join(DATASET_PATH, 'pima-indians-diabetes.data.csv')
dataset = pd.read_csv(data_path, header=None)

# Because thr CSV doesn't contain any header, we add column names
# using the description from the original dataset website
dataset.columns = [
    "NumTimesPrg", "PlGlcConc", "BloodP",
    "SkinThick", "TwoHourSerIns", "BMI",
    "DiPedFunc", "Age", "HasDiabetes"]



# Calculate the median value for BMI
median_bmi = dataset['BMI'].median()
# Substitute it in the BMI column of the
# dataset where values are 0
dataset['BMI'] = dataset['BMI'].replace(
    to_replace=0, value=median_bmi)

# Calculate the median value for BloodP
median_bloodp = dataset['BloodP'].median()
# Substitute it in the BloodP column of the
# dataset where values are 0
dataset['BloodP'] = dataset['BloodP'].replace(
    to_replace=0, value=median_bloodp)

# Calculate the median value for PlGlcConc
median_plglcconc = dataset['PlGlcConc'].median()
# Substitute it in the PlGlcConc column of the
# dataset where values are 0
dataset['PlGlcConc'] = dataset['PlGlcConc'].replace(
    to_replace=0, value=median_plglcconc)

# Calculate the median value for PlGlcConc
median_plglcconc = dataset['PlGlcConc'].median()
# Substitute it in the PlGlcConc column of the
# dataset where values are 0
dataset['PlGlcConc'] = dataset['PlGlcConc'].replace(
    to_replace=0, value=median_plglcconc)

# Calculate the median value for TwoHourSerIns
median_twohourserins = dataset['TwoHourSerIns'].median()
# Substitute it in the TwoHourSerIns column of the
# dataset where values are 0
dataset['TwoHourSerIns'] = dataset['TwoHourSerIns'].replace(
    to_replace=0, value=median_twohourserins)



# Split the training dataset in 80% / 20%
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(
    dataset, test_size=0.2, random_state=42)

# Separate labels from the rest of the dataset
train_set_labels = train_set["HasDiabetes"].copy()
train_set = train_set.drop("HasDiabetes", axis=1)

test_set_labels = test_set["HasDiabetes"].copy()
test_set = test_set.drop("HasDiabetes", axis=1)

train_set = train_set.astype(float)

# Feature Scaling

# Apply a scaler
from sklearn.preprocessing import MinMaxScaler as Scaler

scaler = Scaler()
scaler.fit(train_set)
train_set_scaled = scaler.transform(train_set)
test_set_scaled = scaler.transform(test_set)

# Import all the algorithms we want to test
from sklearn.ensemble import RandomForestClassifier

def sample_loss(params):
    return cross_val_score(
        RandomForestClassifier(
            n_estimators=params[0],
            max_features=params[1],
            min_samples_leaf = params[2],
            random_state=12345),
        X=data,
        y=target,
        scoring='balanced_accuracy',
        cv=5,
        verbose=0).mean()


data = train_set_scaled     # X
target = train_set_labels   # Y

# How many parameters we search
n_params = 3

# Definition bounds for grid and random search
bounds = np.array([[1, 100],                  # n_estimators
                   [1, data.shape[1]],       # max_features
                   [1, 50]])                  # min_samples_leaf

# Distribution values for each parameter
param_dist = {'n_estimators':range(2,10),
                "max_features": range(2, data.shape[1]),
                "min_samples_leaf": range(1, 30),
                }

# GRID SEARCH
print ("---- GRID SEARCH ----")
clf = RandomForestClassifier()
grid = GridSearchCV(estimator=clf, param_grid=param_dist, cv = 5, n_jobs=-1)

start_time = time.time()
grid_result = grid.fit(data, target)
# Summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
print("Execution time: " + str((time.time() - start_time)) + ' ms')

# RANDOM SEARCH
print ("---- RANDOM SEARCH ----")
n_iter_search = 5
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search,
                                   n_jobs=-1,
                                   cv=5,
                                   iid=False)

start = time.time()
random_search.fit(data, target)
print("Best: %f using %s" % (random_search.best_score_, random_search.best_params_))
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time.time() - start), n_iter_search))


# BAYESIAN OPTIMIZATION
param_grid = np.array([[x, y, z]
                       for x in np.random.randint(bounds[0][0], bounds[0][1], size=n_params)
                       for y in np.random.randint(bounds[1][0], bounds[1][1], size=n_params)
                       for z in np.random.randint(bounds[2][0], bounds[2][1], size=n_params)])
start_time = time.time()


# B A Y E S I A N     O P T I M I Z A T I O N
import gp
print ("---- BAYESIAN OPTIMIZATION ----")
start = time.time()

xp, yp = gp.bayesian_optimisation(n_iters=30,
                               sample_loss=sample_loss,
                               bounds=bounds,
                               n_pre_samples=3,
                               random_search=100000)
# print("xp:", xp, "yp:", yp)
# get max index from results
max_idx = [np.array(yp).argmax()]
#print (max_idx)
print (xp[max_idx][0])
print("Best: %s using %s cv: %f" % (max_idx[0], xp[max_idx[0]], yp[max_idx[0]]))
print("Bayesian optimization took %.2f seconds for %d candidates." % ((time.time() - start), len(yp)))





