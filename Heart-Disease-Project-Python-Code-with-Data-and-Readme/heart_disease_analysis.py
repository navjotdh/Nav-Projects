#This project explores demographic and clinical characteristics associated with heart diseases using 
#Cleveland Heart Diseases dataset. 
#Required libraries  pandas, matplotlib, and scikit-learn
#How to run:
    #1. Download the heart_cleveland_upload.csv from https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci?resource=download
    #2. Upload the file path
    #3. Run the program
    
    
#Importing libraries 

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#Loading the heart disease dataset
heart=pd.read_csv('heart_cleveland_upload.csv')
#Using functions and attributes to understand the downloaded data
print(heart.head())
print(heart.tail())
print(heart.describe())
print(heart.info)
print (heart.shape)


#Couting the missing and duplicated rows
print(heart.isnull().sum())
print(heart.duplicated().sum())
# There are no missing values and duplicated rows so no additional data cleaning is requred so moving to next step 


#Counting and printing patients with and without heart disease condition 
#This show how many patients are in each category
#As mentioned in details on Kaggle website 1: disease and 0: no disease
condition_counts = heart["condition"].value_counts()
print("Number of patients with heart disease:", condition_counts[1]) 
print("Number of patients without heart disease:", condition_counts[0])

#Comparing gender and heart condition
# Counting the total number of male and female patients to understand the distribution of patients in the dataset
gender_group = heart["sex"].value_counts()
print("Number of Males", gender_group[1]) 
print("Number of Females:", gender_group[0])

#Creating a cross-tabulation to count males and females with and without heart disease
#This shows whether number of heart disease cases differs between males and females patients
#Sex: 1=male and 0=female
#Condition: 1=disease and 0=no disease
sex_condition = pd.crosstab(heart["sex"],heart["condition"])
print("Number of male patients with heart disease:",sex_condition.loc[1, 1])
print("Number of male patients without heart disease:", sex_condition.loc[1, 0])
print("Number of female patients with heart disease:",sex_condition.loc[0, 1])
print("Number of female patients without heart disease:", sex_condition.loc[0, 0])

#Calculating percentage of patients with heart disease within each group
#Percentages are used because the dataset contains different numbers of male and female patients
male_disease_rate = sex_condition.loc[1, 1] / gender_group[1]*100
female_disease_rate = sex_condition.loc[0, 1] / gender_group[0]*100
print("Percent of male patients with heart disease:", str(round(male_disease_rate, 2)) + "%")
print("Percent female patients with heart disease:",str( round(female_disease_rate, 2)) + "%")

#Creating a bar chart showing the proportion of patients with heart disease by gender
sex_disease_rate = pd.Series({"Female": female_disease_rate,
"Male": male_disease_rate})

# Creating the bar chart
sex_disease_rate.plot(kind="bar")
plt.title("Percentage of Patients with Heart Disease by Sex")
plt.ylabel("Percentage (%)")
# Adjusting the layout so labels are not cut off
plt.tight_layout()
# Saving the figure 
plt.savefig("heart_disease_percent_by_sex.png")
plt.show()

#Comparing age, gender and heart condition
#Grouping patients by sex and heart condition and summarzing by age column to find
#count (number of patients in each group), mean (average age in each group), and median
#This shows whether age patterns differ between male and female patients with and without heart disease
#sex (1:male, 0: female) and heart condition (1: disease and 0: no disease)
age_sex_condition = heart.groupby(["sex", "condition"])["age"].agg(
["count", "mean", "median"]) #Splitting the dataset 
print(age_sex_condition.round(2))

#Creating a histogram to show how patient ages are distributed
heart["age"].plot(kind="hist",bins=10)
plt.title("Distribution of Patient Age")
plt.xlabel("Age in Years")
plt.ylabel("Number of Patients")
# Adjusting the layout so labels are not cut off
plt.tight_layout()
# Saving the histogram
plt.savefig("patient_age_distribution.png")
plt.show()


# Grouping patients by exercise-induced angina and heart condition and summarizing the oldpeak values
# and summarizing ST depression values (1: yes exercise induced angina and 0: not excercise induced angina)
# This helps determine whether ST depression differ according to exercise-induced angina and heart disease condition.
stress_test_analysis = heart.groupby(["exang", "condition"])["oldpeak"].agg(["count", "mean", "median", "std"])
print(stress_test_analysis.round(2))

# Comparing chest pain type with heart disease
# Creating a cross tabulation that counts patients in each chest pain and heart disease category.
#This help determine whether heart disease occurs more frequently within certain chest pain categories
cp_condition = pd.crosstab(heart["cp"],heart["condition"])
print(cp_condition)
#cp: chest pain type: 
#Value 0: typical angina, Value 1: atypical angina, Value 2: non-anginal pain, Value 3: asymptomatic
# 1: disease and 0: No disease
# Creating a bar chart to compare chest pain type with heart disease condition
cp_condition.plot(kind="bar")
plt.title("Chest Pain Type by Heart Disease condition")
plt.xlabel("Chest Pain Type:  0 = Typical, 1 = Atypical, 2 = Non-anginal, 3 = Asymptomatic")
plt.ylabel("Number of Patients")
plt.legend(["No Heart Disease", "Heart Disease"])
# Adjusting the layout so labels are not cut off
plt.tight_layout()
# Saving the chart 
plt.savefig("chest_pain_by_heart_condition.png")
plt.show()

# Calculating average maximum heart rate by heart condition
# This will help determine whether patients with heart disease had a different average max heart rate than patients without heart condition
heart_rate_by_condition = heart.groupby("condition")["thalach"].mean()
print("Average maximum heart rate of patients with heart disease:", round(heart_rate_by_condition[1], 2))
print("Average maximum heart rate of patients without heart disease:", round(heart_rate_by_condition[0], 2))

# Finding average resting blood pressure by heart disease condition
# to determine whether it differs between the two patient groups
blood_pressure_by_condition = heart.groupby("condition")["trestbps"].mean()
print("Average resting blood pressure of patients with heart disease:", round(blood_pressure_by_condition[1], 2))
print("Average resting blood pressure of patients without heart disease:", round(blood_pressure_by_condition[0], 2))

# Finding average cholesterol by heart disease condition to see if cholesterol level differ between patient with and without heart disease?
chol_by_condition = heart.groupby("condition")["chol"].mean()
print("Average cholesterol of patients with heart disease:", round(chol_by_condition[1], 2))
print("Average cholesterol of patients without heart disease:",round(chol_by_condition[0], 2))



#Creating a list of predictor variables to help classify patients as with or without heart disease
predictors = ["age","sex","cp","trestbps", "chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]

#Dividing the dataset into training and testing portions
#70% of data is used for training and 30% for testing
#random_state makes the split reproducible
#stratify keeps a similar proportion of heart disease cases in training and testing data
X = heart[predictors]
y = heart["condition"]

x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=42,stratify=y)


#Checking the shapes
print (x_train.shape)
print (x_test.shape)
print (y_train.shape)
print (y_test.shape)
# Creating the logistic regression model
model = LogisticRegression(max_iter=2000)
# Training the model using the training data
model.fit(x_train, y_train)
# Using the trained model to predict the testing outcomes
y_predict = model.predict(x_test)


#comparing the acutal heart disease outcomes with the predicted outcomes
print("Actual heart disease outcomes:", y_test[:10])
print("Predicted heart disease outcomes:", y_predict[:10])
# Calculating the accuracy percentage of the logistic regression model
accuracy = accuracy_score(y_test, y_predict)
print('\nModel accuracy: ', str(round(accuracy * 100, 2)) + '%')


