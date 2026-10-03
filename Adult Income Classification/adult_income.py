#Adult Income Classification Using Machine Learning
#This project uses the Adult Income dataset to examine how well education,
#race, sex, and relationship status can classify whether an individual's
#annual income is above $50,000.
#Logistic Regression and Random Forest models are developed and compared
# using model evaluation, cross-validation, hyperparameter tuning, feature
# importance, and demographic performance checks.


#Importing libraries
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import cross_val_score  
from sklearn.model_selection import GridSearchCV 

#Loading the dataset
df = pd.read_csv('Adult Income Classification/adult.csv') 
print(df.head(5)) 
print(df.columns) 
print(df.shape) 
 
#Replacing '?' with missing values 
df = df.replace('?', pd.NA) 

#Checking for missing values and duplicates
print('Missing values by variable: \n', df.isnull().sum()) 
print('Duplicates: ', df.duplicated().sum()) 
 
#Dropping duplicates 
df = df.drop_duplicates() 

#Printing shape after dropping duplicates 
print(df.shape) 
 
#Summarizing missing values by variable 
missing_summary = pd.DataFrame({'Missing Count': df.isnull().sum(), 'Missing Percentage': df.isnull().mean() * 100}) 
print(missing_summary[missing_summary['Missing Count'] > 0].sort_values('Missing Percentage')) 
 
#Replacing missing categorical values with 'Unknown' 
df['native.country'] = df['native.country'].fillna('Unknown') 
df['workclass'] = df['workclass'].fillna('Unknown') 
df['occupation'] = df['occupation'].fillna('Unknown') 

#Checking for missing values again 
print(df.isnull().sum()) 
print((df.isnull().sum()).sum()) 
 
#Checking value distributions  
print(df['education'].value_counts()) 
print(df['race'].value_counts()) 
print(df['sex'].value_counts()) 
print(df['relationship'].value_counts()) 
print(df['income'].value_counts()) 

#Calculating income distribution as percentages
income_percent = df['income'].value_counts(normalize=True) * 100 
print(income_percent) 
 
#Research question 
#How well can education, race, sex, and relationship status predict whether an individual's income is above $50,000? 
 
#Defining x and y  
x = df[['education', 'race', 'sex', 'relationship']] 
y = df['income'] 
 
#Converting categorical variables into numeric dummy variables  
x = pd.get_dummies(x, drop_first=True, dtype=int) 
print(x.head()) 
print(x.columns) 
print(x.shape) 
 
#Splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y) 
 
#Initializing the Logistic Regression model 
model = LogisticRegression() 
 
#Training the model  
model.fit(x_train, y_train) 
 
#Predicting the classes 
y_pred = model.predict(x_test) 
 
#Evaluating the Logistic Regression model
accuracy_s = accuracy_score(y_test, y_pred) 
confusion_m = confusion_matrix(y_test, y_pred) 
classification_r = classification_report(y_test, y_pred) 
 
print('Logistic Regression Model Results') 
print('Accuracy score:', accuracy_s) 
print('Confusion Matrix \n', confusion_m) 
print('Classification Report \n', classification_r) 
 
#Initializing the Random Forest model 
rf_model = RandomForestClassifier(random_state=42) 
 
#Training the model 
rf_model.fit(x_train, y_train) 
 
#Predicting the classes 
rf_pred = rf_model.predict(x_test) 
 
#Evaluating the Random Forest model
rf_accuracy = accuracy_score(y_test, rf_pred) 
rf_confusion = confusion_matrix(y_test, rf_pred) 
rf_report = classification_report(y_test, rf_pred) 
 
print('Random Forest Model Results') 
print('Accuracy Score:', rf_accuracy) 
print('Confusion Matrix:') 
print(rf_confusion) 
print('Classification Report:') 
print(rf_report) 
 
#Performing 5-fold cross-validation for Logistic Regression 
cv_scores = cross_val_score(model, x_train, y_train, cv=5, scoring='accuracy') 
print('Logistic Regression Cross Validation Scores:', cv_scores) 
print('Average Logistic Regression CV Accuracy:', cv_scores.mean()) 
 
#Performing 5-fold cross-validation for Random Forest 
rf_cv_scores = cross_val_score(rf_model, x_train, y_train, cv=5, scoring='accuracy') 
print('Random Forest Cross Validation Scores:', rf_cv_scores) 
print('Average Random Forest CV Accuracy:', rf_cv_scores.mean()) 
 
#Setting parameter values for Logistic Regression 
logistic_para_grid = {'C': [0.1, 1, 10]} 
 
#Performing GridSearchCV for Logistic Regression 
logistic_grid = GridSearchCV(LogisticRegression(), logistic_para_grid, cv=5, scoring='accuracy') 
 
#Training the model with different parameter values 
logistic_grid.fit(x_train, y_train) 
 
#Checking the best parameter and score 
print('Best Logistic Regression parameter:', logistic_grid.best_params_) 
print('Best Logistic Regression CV Accuracy:', logistic_grid.best_score_) 
 
#Setting parameter values for Random Forest 
rf_param_grid = {'n_estimators': [50, 100, 200]} 
 
#Performing GridSearchCV for Random Forest 
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=5, scoring='accuracy') 
 
#Training the model with different parameter values 
rf_grid.fit(x_train, y_train) 
 
#Checking the best parameter and score 
print('Best Random Forest Parameter', rf_grid.best_params_) 
print('Best Random Forest CV Accuracy', rf_grid.best_score_) 
 
#Getting the best models from GridSearchCV 
best_logistic_model = logistic_grid.best_estimator_ 
best_rf_model = rf_grid.best_estimator_ 
 
#Making predictions with the tuned models 
best_logistic_pred = best_logistic_model.predict(x_test) 
best_rf_pred = best_rf_model.predict(x_test) 
 
#Evaluating the tuned Logistic Regression model 
print('Tuned Logistic Regression Accuracy:', accuracy_score(y_test, best_logistic_pred)) 
print('Tuned Logistic Regression Classification Report:') 
print(classification_report(y_test, best_logistic_pred)) 
 
#Evaluating the tuned Random Forest model 
print('Tuned Random Forest Accuracy:', accuracy_score(y_test, best_rf_pred)) 
print('Tuned Random Forest Classification Report:') 
print(classification_report(y_test, best_rf_pred)) 
 
#Checking Random Forest feature importance 
feature_importance = pd.DataFrame({'Feature': x.columns, 'Importance': best_rf_model.feature_importances_}) 
feature_importance = feature_importance.sort_values(by='Importance', ascending=False) 
 
print('Random Forest Feature Importance:') 
print(feature_importance) 
 
#Creating a dataframe for demographic performance checks 
ethical_check = df.loc[x_test.index, ['race', 'sex']].copy() 
ethical_check['actual'] = y_test 
ethical_check['predicted'] = best_logistic_pred 
ethical_check['correct'] = ethical_check['actual'] == ethical_check['predicted'] 
 
#Checking Logistic Regression accuracy by sex 
print('Accuracy by Sex:') 
print(ethical_check.groupby('sex')['correct'].mean()) 
 
#Checking Logistic Regression accuracy by race 
print('Accuracy by Race:') 
print(ethical_check.groupby('race')['correct'].mean())