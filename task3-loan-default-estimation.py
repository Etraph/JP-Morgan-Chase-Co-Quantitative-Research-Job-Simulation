import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
import seaborn as sns

'''
This code uses "Loan_Data.csv", described below. To run the code, one has to use a dataframe containing at least 'customer_id', 'default' (value 0 or 1), 'loan_amt_outstanding', that is the customer's loan amount oustanding.

For a set a 10,000 customer IDs, the example data used provides in order information on income, credit lines outstanding, loan amount outstanding, total debt outstanding, income, years employed, fico score and default.
To estimate a customer's probability of default with this data, we first split it into features (dropping the IDs) and targets (0-1 default), before separating the features into a test set and a training set, using an 80%/20% ratio.
It is important to standardize the featured data because, we work with different units, especially when comparing credit lines, their values ranging from 0 to 10,  with income reaching ~100.000 values, and even more in other contexts. 
'''

#--------------------------------------------------------
# Data initialisation and train/test samples preparation
#--------------------------------------------------------

loandata = pd.read_csv("Loan_Data.csv")
X = loandata.drop(['default','customer_id'],axis=1)        # Features
scaleX = StandardScaler().fit_transform(X)
Y = loandata['default']        # Targets

# Split data to train on 70% of the featured data
trainX, testX, trainY, testY = train_test_split(scaleX,Y,test_size = 0.3,random_state = 42)         #=42 to ensure reproducibility of the results

#--------------------
#Logistic Regression
#--------------------

'''
A logistic regression seems to be the soundest strategy because we are trying to predict a result that has binary outcomes.
'''

loan_amounts = X['loan_amt_outstanding']
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)        # Cross-validation setup

def expected_loss_absolute(model, X, Y, loan_amounts, lgd=0.9):
  '''
  input: - model: ML model used on the prepared data
         - X: feature series
         - Y: target series
         - loan_amounts: loan amount series
  output: absolute expected loss over the training set
  '''
    pd_preds = model.predict_proba(X)[:, 1]
    expected_losses = pd_preds * lgd * loan_amounts
    real_losses = Y * lgd * loan_amounts
    absolute_errors = np.abs(expected_losses - real_losses)
    return absolute_errors.mean()


def objective(trial):
  '''
  input: current trial set for the model
  output: average expected loss over the trial
  return the objective function to minimise
  '''
    C = trial.suggest_float('C', 0.01, 100, log=True)  # zooming into the range of interest
    penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])  # still tune l1 vs l2
    solver = 'liblinear' if penalty == 'l1' else 'lbfgs'
    
    model = LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=1000, random_state=42)

    scores = []
    for train_idx, valid_idx in cv.split(scaleX, Y):
        trainX, validX = scaleX[train_idx], scaleX[valid_idx]
        trainY, validY = Y.iloc[train_idx], Y.iloc[valid_idx]
        trainLoans, validLoans = loan_amounts.iloc[train_idx], loan_amounts.iloc[valid_idx]

        m = clone(model)
        m.fit(trainX, trainY)
        
        score = expected_loss_absolute(m, validX, validY, validLoans)
        scores.append(score)
    
    return np.mean(scores)

# Run Auto ML algorithm
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

# Best model output
print("Best Logistic Regression:")
print(study.best_trial.params)
print(f"Average Absolute Expected Loss: {study.best_trial.value:.2f} EUR")

logreg = LogisticRegression(C=100)        # Initialising with logreg parameter C, which after implementation turns out to improve results as it tends to infinity
logreg.fit(trainX,trainY)         # Fitting the training data

# Predicted labels for the test sample
predictedY = logreg.predict(testX)
# Confusion matrix
confusion = confusion_matrix(testY,predictedY)

# Confusion matrix plot
plt.figure(figsize=(8,6))
sns.heatmap(confusion, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
plt.title('Confusion Matrix')
plt.show()

'''
The example data yields the following performance:
    precision    recall  f1-score   support

           0       1.00      1.00      1.00      2457
           1       1.00      0.99      1.00       543

    accuracy                           1.00      3000
   macro avg       1.00      1.00      1.00      3000
weighted avg       1.00      1.00      1.00      3000

One can also access the features that were most indicative of the credit default.
'''
feature_names = X.columns
coefficients = logreg.coef_[0]
coeff = pd.DataFrame({'Features': feature_names, 'Coefficients': coefficients})

coeff.sort_values(by = 'Coefficients', ascending=False)

'''
Features	Coefficients
0	credit_lines_outstanding	37.761533
2	total_debt_outstanding	13.198661
1	loan_amt_outstanding	1.236759
5	fico_score	-4.522371
3	income	-9.037624
4	years_employed	-12.767514
This is surprising because the ficoscore and income would seem to be the most important as well when trying to predict credit default.
'''
#------------------------
# Probability of default
#------------------------

'''
To conclude, we compute the expected loss for each individual as:
$$
\text{Expected Loss} = \text{total debt outstanding}*\text{proba default}*0.9
$$
'''

proba_default = logreg.predict_proba(scaleX)[:,1]
loandata['expected_loss'] = proba_default*loandata['total_debt_outstanding']*0.9
loandata['expected_loss'] = loandata['expected_loss'].apply(lambda x: 0 if abs(x) < 0.01 else round(x, 2))

loandata['real_loss'] = 0.9*loandata['total_debt_outstanding']*loandata['default']

mean_error = expected_loss_absolute(logreg, scaleX, Y, loan_amounts, lgd=0.9)
print(mean_error)

'''
yields with the example data 9.762478569459258, that is a 0.1% error
One important remark: it seems the optimization of hyperparameter $C$ yields a better mean error as the range for $C$ increases.
One could thus assert that a logistic regression model without penalty, that is $C = \infty$, would yield the best result under the condition that there is no overfitting.
'''
