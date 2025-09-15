# SBA Loan Default Prediction - Machine Learning Approach

## Libraries
pandas, numpy, scikit-learn, matplotlib, seaborn, statsmodels, dmba, mord

## Project Overview
Predictive modeling of Small Business Administration (SBA) loan defaults using California loan data.  
Models are trained and evaluated to classify loans as paid in full or default.

## Data
 Preprocessing: dropped identifiers, recoded categorical fields, removed missing values, sampled 5,000 records

## Workflow
1. **Preprocessing**: handle NAs, feature scaling, categorical encoding  
2. **Train-Test Split**: stratified sampling for balanced classes  
3. **Models**: Logistic Regression, kNN, Decision Tree, Bagging, Random Forest, Gradient Boosting, AdaBoost, Neural Network (MLP), LDA, Ordinal Logistic Regression  
4. **Evaluation**: Accuracy, Precision, Recall, F1, ROC/AUC, Confusion Matrix, Gains & Lift Charts  
5. **Tuning**: GridSearchCV for kNN, Random Forest, Gradient Boosting, and MLP  

## Results
- Models compared across accuracy and AUC  
- Best model identified based on classification performance
