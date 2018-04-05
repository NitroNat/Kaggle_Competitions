#! python3
""" Kaggle Titanic Dataset
Reference: https://www.kaggle.com/c/titanic

History:
v1. Simple Holdout testing using Random Forest 76.555%
v2. Replaced simple holdout with 5-fold CV, picking non-linear SVM 76.555%
v3. Add Hyperparameter tuning for non-linear SVM using Nested CV 79.425%
"""
__author__ = "NF"

import os
import sys
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns

# File I/O
# Load the training and testing set
curr_dir = os.getcwd()
data_dir = os.path.join(curr_dir, "titanic", "data")
train_data_path = os.path.join(data_dir, 'train.csv')
test_data_path = os.path.join(data_dir, 'test.csv')
# Import the data
# Pandas is the most convenient for csv files (tabular data)
# The training set is for our test/validation set
train_df = pd.read_csv(train_data_path)
# Do not touch the test set until the final prediction for submission
test_df = pd.read_csv(test_data_path)

# Understand the data
# List the features
print(train_df.columns.values)
print(test_df.columns.values)
# Determine the dimension of the data
print(train_df.shape)
# *Obs: "Survived" feature is the target
print(test_df.shape)

# Peek at data
test_df.head(n=5)
test_df.tail(n=5)
train_df.head(n=5)
train_df.tail(n=5)
# *Obs: age can be float
# What are the feature data types?
train_df.info(verbose=None, buf=None, max_cols=None,
                            memory_usage=None,
                            null_counts=None)
test_df.info()
# *Obs: it looks like Pclass, SibSp, Parch should be categorical
# Correct the data types
'''
print(set(train_df["Pclass"]))
print(set(train_df["SibSp"]))
print(set(train_df["Parch"]))

feats_obj_lst = train_df.select_dtypes(include=[np.object]).columns
for feature in list(feats_obj_lst) + ['Pclass','SibSp','Parch', 'Survived']:
    train_df[feature] = train_df[feature].astype('category')

train_df.info()
'''
# Quick Summary of the data
# Numeric features
train_df.describe(percentiles=None, include=[np.number], exclude=None)
# Categorical features
train_df.describe(percentiles=None, include=[np.object], exclude=None)

feats_numeric_lst = list(train_df.select_dtypes(include=[np.number]).columns)
feats_categorical_lst = list(train_df.select_dtypes(include=[
    'category']).columns)

# Any Missing Values?
print(np.any(pd.isnull(train_df)))
print(pd.isnull(train_df).sum())
# obs: Age, Cabin, Embarded missing values


# For each feat, what proportion of Survivors?
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)

g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)

# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();

# grid = sns.FacetGrid(train_df, col='Embarked')
grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()

# grid = sns.FacetGrid(train_df, col='Embarked', hue='Survived', palette={0: 'k', 1: 'w'})
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()

combine = [train_df, test_df]
print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

# Remove these features
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

print("After", train_df.shape, test_df.shape, combine[0].shape, combine[
    1].shape)

################################################################################
# Feature Engineering
################################################################################
# Create "Titles" feature from names
# Look at the names for their titles to see
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_df['Title'], train_df['Sex'])

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', \
                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train_df.head()

# Don't need the name anymore or passenger ID since they are unique for each
# sample
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
train_df.shape, test_df.shape

# to category
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train_df.head()

# Age Imputation
# grid = sns.FacetGrid(train_df, col='Pclass', hue='Gender')
grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()

guess_ages = np.zeros((2,3))
guess_ages

for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                               (dataset['Pclass'] == j + 1)]['Age'].dropna()
            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5

    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j + 1), \
                        'Age'] = guess_ages[i, j]

    dataset['Age'] = dataset['Age'].astype(int)

train_df.head()

# Bin the Age into age groups
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
for dataset in combine:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
train_df.head()

train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]
train_df.head()

# Add Feature Size of the family
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)

for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()

train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]

train_df.head()

# Age group * social class
for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)

# Fix embarked
freq_port = train_df.Embarked.dropna().mode()[0]
freq_port

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived',
                                                                                            ascending=False)

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train_df.head()

# Fix Fare
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
test_df.head()

train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)

for dataset in combine:
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]

train_df.head(10)
test_df.head(10)

# review feature correlations
coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])
coeff_df.sort_values(by='Correlation', ascending=False)

################################################################################
# Training Model
################################################################################
# Prep the training and testing sets
X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, \
    BaggingClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from time import time
estimators = [('NaiveBayes', make_pipeline(StandardScaler(), GaussianNB())),
              ('GaussianProcess', GaussianProcessClassifier()),
                ('Logistic', make_pipeline(StandardScaler(), SGDClassifier(loss='log'))),
                ('KNN', make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))),
                ('LDA', LinearDiscriminantAnalysis()),
                ('QDA', QuadraticDiscriminantAnalysis()),
                ('decision_tree', DecisionTreeClassifier()),
                ('linear_svc', make_pipeline(StandardScaler(), LinearSVC())),
                ('SVM', make_pipeline(StandardScaler(), SVC(kernel='rbf'))),
                ('RF', RandomForestClassifier(n_estimators=100)),
                ('Extreme', ExtraTreesClassifier(n_estimators=50)),
                ('Adaboost', AdaBoostClassifier()),
                ('GradientBoosting', GradientBoostingClassifier()),
                ('BaggingClassifier', BaggingClassifier()),
                ('XGB', XGBClassifier()),
                ('MLP', make_pipeline(StandardScaler(), MLPClassifier(solver='lbfgs',
                                                                    alpha=1e-5,
                                                                    hidden_layer_sizes=(30, 1),
                                                                    random_state=1))),
                ('perceptron', Perceptron()),
              ]

from sklearn.model_selection import cross_validate, StratifiedKFold
perf_metrics = ['accuracy', 'f1']
cv_scheme = StratifiedKFold(n_splits=5, shuffle=False, random_state=None)

scores = {}
for name, estimator in estimators:
    start = time()
    print(f'Training: {name}')
    perf = cross_validate(estimator=estimator,
                             X=X_train, y=Y_train, scoring=perf_metrics,
                             cv=cv_scheme, return_train_score=False)
    end = time()
    scores[name] = np.mean(perf["test_accuracy"])
    print(f'Time elapsed: {end-start}')


models = pd.DataFrame.from_dict(data=scores, orient='index')
models.columns = ['Accuracy']
models.sort_values(by='Accuracy', ascending=False)

# Choose the best classifier and tune parameters: SVM
# Nested Cross Validation with GridSearch
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

parameters = [{'gamma': np.geomspace(1e-3, 1e3, 10,dtype=np.float32),
                'C': np.geomspace(1e-3, 1e3, 10,dtype=np.float32)}]
# Inner CV
clf = GridSearchCV(SVC(), param_grid= parameters, scoring='accuracy', cv=5)
cv_scheme = StratifiedKFold(n_splits=5, shuffle=True)
my_scaler = StandardScaler()
best_parameters = []
scores = []
all_y_pred = np.array([], dtype=np.uint32)
all_y_true = all_y_pred.copy()

n_trials = 5
for n in range(2):
    # Outer CV
     for train_index, test_index in cv_scheme.split(X=np.zeros(X_train.shape),
                                                    y=Y_train):
        X_train_scaled = my_scaler.fit_transform(X=X_train.iloc[train_index,:])
        clf.fit(X=X_train_scaled, y=Y_train[train_index])
        best_parameters.append(clf.best_params_)
        print(clf.best_params_)

        X_test_scaled = my_scaler.transform(X_train.iloc[test_index,:])
        y_pred = clf.best_estimator_.predict(X_test_scaled)

        all_y_pred = np.concatenate((all_y_pred, y_pred))
        y_true = Y_train[test_index]
        all_y_true = np.concatenate((all_y_true, y_true))

        score = accuracy_score(y_true, y_pred)
        print(score)
        scores.append(score)
print(np.mean(scores))

# Make final predictions with the most consistent classifier
final_clf = make_pipeline(StandardScaler(), SVC(kernel='rbf',
                                                **best_parameters[-2]))
final_clf.fit(X_train, Y_train)
Y_pred = final_clf.predict(X_test)

submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })

submission.to_csv('./titanic/submission.csv', index=False)




