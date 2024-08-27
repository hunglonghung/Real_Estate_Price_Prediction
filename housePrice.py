import pickle
import pandas as pd
import numpy as np
from sklearn.svm import SVC,SVR
from sklearn.metrics import mean_squared_error, r2_score
# from ydata_profiling import ProfileReport
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
df = pd.read_csv('Real estate.csv')
# data visualization 
    # profile = ProfileReport(df,title = 'Real Estate Report',explorative = True)
    # profile.to_file('HousePrice.html')

# checking null values
print(df.info())
print(df.isnull().sum())

# separating datasets
target = 'Y house price of unit area'
X = df.drop(labels=target,axis=1)
Y = df[target]

# Split dataset, since there is no null values to handle
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
X_test,X_val,Y_test,Y_val = train_test_split(X_test,Y_test,test_size= 0.5,random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Cross validation for best Value
params = {
    'C':[1,5,10,20,40,80,160],
    'kernel':['linear','poly','rbf','sigmoid'],
}
params_poly = {
    'C':[1,5,10,20,40,80,160],
    'kernel':['poly'],
    'degree':[2,3,4,5]
}
clf = GridSearchCV(
    estimator = SVR(),
    param_grid = [params,params_poly],
    cv = 4,
    verbose = 1,
    scoring = 'r2',
    n_jobs=-1
)
clf.fit(X_train,Y_train)
Y_pred = clf.predict(X_test)
with open('house_price_SVR_model.pkl','wb') as file:
    pickle.dump(clf,file)
print(clf.best_score_)
print(clf.best_params_)
print(f"Best R^2 Score: {clf.best_score_}")
print(f"Best Parameters: {clf.best_params_}")
print(f"Mean Squared Error: {mean_squared_error(Y_test, Y_pred)}")
print(f"R^2 Score: {r2_score(Y_test, Y_pred)}")



