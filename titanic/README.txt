Kaggle Titanic Dataset

About:
Ccomplete the analysis of what sorts of people were likely to survive.
In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.

Goal:
Predict if a passenger survived the sinking of the Titanic or not.
For each PassengerId in the test set, you must predict a 0 or 1 value for the Survived variable.

Evalutaion:
Your score is the percentage of passengers you correctly predict. This is known simply as "accuracy”.
You should submit a csv file with exactly 418 entries plus a header row. Your submission will show an error if you have extra columns
(beyond PassengerId and Survived) or rows.

Submission Format:
1) PassengerId (sorted in any order)
2) Survived (contains your binary predictions: 1 for survived, 0 for deceased)
E.g.
----------------------------
PassengerId,Survived
 892,0
 893,1
 894,0
 Etc.
-----------------------------
================================================================================
Features:

Variable	Definition	    Key
survival	Survival	    0 = No, 1 = Yes
pclass	    Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
sex	        Sex
Age	        Age in years
sibsp	    # of siblings / spouses aboard the Titanic
parch	    # of parents / children aboard the Titanic
ticket	    Ticket number
fare	    Passenger fare
cabin	    Cabin number
embarked	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton

Variable Notes
pclass: A proxy for socio-economic status (SES)
1st = Upper
2nd = Middle
3rd = Lower

age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
sibsp: The dataset defines family relations in this way...
Sibling = brother, sister, stepbrother, stepsister
Spouse = husband, wife (mistresses and fiancés were ignored)

parch: The dataset defines family relations in this way...
Parent = mother, father
Child = daughter, son, stepdaughter, stepson
Some children travelled only with a nanny, therefore parch=0 for them.

Reference: https://www.kaggle.com/c/titanic
"""