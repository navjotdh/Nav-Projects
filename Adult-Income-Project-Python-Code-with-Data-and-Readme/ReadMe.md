# Income Classification Using Machine Learning

## Project Overview
This project uses the Adult Income dataset to examine whether education, race, sex, and relationship status can help classify whether an individual earns more than $50,000 per year. Two machine learning models, Logistic Regression and Random Forest, were developed and compared using Python and scikit-learn.

The project demonstrates data preparation, classification modeling, model evaluation, cross-validation, hyperparameter tuning, feature importance, and an examination of model performance across demographic groups.

## Dataset
The Adult Income dataset originally contained 32,561 records and 15 variables. Data preparation included identifying missing values, removing 24 duplicate records, and handling missing categorical values. After duplicate removal, the dataset contained 32,537 records.

Four variables were selected as predictors:
- Education
- Race
- Sex
- Relationship status

Annual income was used as the target variable and was classified as `<=50K` or `>50K`. Approximately 75.91% of the records belonged to the `<=50K` class and 24.09% belonged to the `>50K` class.

## Machine Learning Approach
The categorical predictors were converted into numerical dummy variables, resulting in 25 features for model development. The dataset was divided into 70% training data and 30% testing data using stratification to maintain the income distribution across both sets.

Two classification models were developed:
- Logistic Regression
- Random Forest

Model performance was evaluated using accuracy, precision, recall, F1-score, confusion matrices, and five-fold cross-validation. GridSearchCV was also used for hyperparameter tuning.

## Results
Logistic Regression achieved 81.85% accuracy, while Random Forest achieved 81.86% accuracy on the test data.

Five-fold cross-validation produced similar results:
- Logistic Regression: 82.06% average accuracy
- Random Forest: 81.84% average accuracy

Although both models achieved approximately 82% overall accuracy, performance differed between the two income classes. Logistic Regression achieved 95% recall for the `<=50K` class but only 42% recall for the `>50K` class. Random Forest showed a similar pattern, with 94% recall for the `<=50K` class and 43% recall for the `>50K` class.

These results demonstrate why overall accuracy should not be considered alone when evaluating a classification model, particularly when the target classes are unevenly distributed.

## Hyperparameter Tuning
GridSearchCV was used to evaluate different model settings. For Logistic Regression, `C = 10` was selected as the best parameter, producing a cross-validation accuracy of approximately 82.09%.

Hyperparameter tuning resulted in only small changes in model performance, indicating that adjusting the tested parameters did not substantially improve classification accuracy.

## Ethical Considerations
Race and sex are sensitive demographic characteristics, making responsible model evaluation particularly important. Model performance was therefore examined across demographic groups in addition to overall performance.

Differences in accuracy were observed across sex and race groups. These differences alone do not establish whether a model is fair or unfair, but they highlight the importance of examining subgroup performance and the use of sensitive characteristics before applying machine learning models to high-impact areas such as employment or lending.


## Data Source
Adult Census Income dataset from Kaggle: https://www.kaggle.com/datasets/uciml/adult-census-income