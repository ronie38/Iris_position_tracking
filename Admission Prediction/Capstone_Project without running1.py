import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
%matplotlib inline
# Loading the dataset
df = pd.read_csv(r'F:\IIIT\Capstone Project I\admission_predict.csv')
print (df)
# Returns number of rows and columns of the dataset
df.shape

df.head()
df.tail()
df.columns
df.info()
df.info()
df.describe().T
df.dtypes
df.isnull().any()
df = df.rename(columns={'GRE Score': 'GRE', 'TOEFL Score': 'TOEFL', 'Chance of Admit': 'Prob'})
df.head()
fig = plt.hist(df['TOEFL'], rwidth=0.7)
plt.title('Distribution of TOEFL Scores')
plt.xlabel('TOEFL Scores')
plt.ylabel('Count')
plt.show()
fig = plt.hist(df['GRE'], rwidth=0.7)
plt.title('Distribution of GRE Scores')
plt.xlabel('GRE Scores')
plt.ylabel('Count')
plt.show()
# Visualizing the feature TOEFL
fig = plt.hist(df['University Rating'], rwidth = 0.7)
plt.title('Distribution of University Rating')
plt.xlabel('University Rating')
plt.ylabel('Count')
plt.show()
# Visualizing the feature TOEFL
fig = plt.hist(df['SOP'], rwidth=0.7)
plt.title('Distribution of SOP')
plt.xlabel('SOP Rating')
plt.ylabel('Count')
plt.show()
# Visualizing the feature TOEFL
fig = plt.hist(df['LOR'], rwidth=0.7)
plt.title('Distribution of LOR Rating')
plt.xlabel('LOR Rating')
plt.ylabel('Count')
plt.show()
# Visualizing the feature TOEFL
fig = plt.hist(df['CGPA'], rwidth=0.7)
plt.title('Distribution of CGPA')
plt.xlabel('CGPA')
plt.ylabel('Count')
plt.show()
# Visualizing the feature TOEFL
fig = plt.hist(df['Research'], rwidth=0.7)
plt.title('Distribution of Research Papers')
plt.xlabel('CGPA')
plt.ylabel('Count')
plt.show()
# Removing the serial no, column
df.drop('Serial No.', axis='columns', inplace=True)
df.head()
# Replacing the 0 values from ['GRE', 'TOEFL','University Rating','SOP','LOR', 'CGPA'] by NaN
df_copy = df.copy(deep=True)
df_copy[['GRE','TOEFL', 'University Rating','SOP', 'LOR', 'CGPA']]= df_copy[['GRE','TOEFL','University Rating','SOP','LOR','CGPA']].replace(0, np.NaN)
df_copy.isnull().sum()
# Splitting the dataset in features and label
x = df_copy.drop('Probability', axis='columns')
y = df_copy['Probability']
# Using GridSearchCV to find the best algorithm for this problem
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
def find_best_model(x, y):
    models = {
        'linear_regression':{
            'model': LinearRegression(),
            'parameters':{
                'normalize':[True, False]
            }
        },
        'lasso':{
            'model':Lasso(),
            'parameters':{
                'alpha':[1,2],
                'selection':['random','cyclic']
            }
        },
        'svr': {
            'model': SVR(),
            'parameters':{
                'gamma':['auto','scale']
            }
        },
        'decision_tree':{
            'model': DecisionTreeRegressor(),
            'parameters':{
                'criterion': ['mse','friedman_mse'],
                'splitter': ['best','random']
            }
        },
        'random_forest':{
            'model': RandomForestRegressor(criterion='mse'),
            'parameters':{
                'n_estimators':[5,10,15,20]
            }
        },
        'knn': {
            'model': KNeighborsRegressor(algorithm='auto'),
            'parameters':{
                'n_neighbors':[2,5,10,20]
            }
        }
    }
    scores = []
    for model_name, model_params in models.items():
        gs = GridSearchCV(model_params['model'], model_params['parameters'],cv = 5, return_train_score = False)
        gs.fit(x,y)
        scores.append({
            'model':model_name,
            'best_parameters': gs.best_params_,
            'score': gs.best_score_
        })
    return pd.DataFrame(scores, columns = ['model','best_parameters','score'])
find_best_model(x,y)
# Using cross_val_score for gaining highest accuracy
from sklearn.model_selection import cross_val_score
scores = cross_val_score(LinearRegression(normalize = True), x, y, cv = 5)
print('Highest Accuracy':{}%.format(round(sum(scores)*100/len(scores)),3))
#Splitting the dataset into train and test samples
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.20, random_state = 5)
print(len(x_train), len(x_test))
# Creating the Linear Regression Model
model = LinearRegression(normalize=True)
model.fit(x_train,y_train)
model.score(x_test, y_test)
# Prediction 1
# Input in the form : GRE, TOEFL, University Rating, SOP, LOR, CGPA, Research
print('Chance of getting into UCLA is{}%'.format(round(model.predict([[337,118,4,4.5,9.65,0]])[0]*100,3)))
# Prediction 2
# Input in the form : GRE, TOEFL, University Rating, SOP, LOR, CGPA, Research
print('Chance of getting into UCLA is{}%'.format(round(model.predict([[320,113,2,2.0,2.5,8.64,1]])[0]*100,3)))

