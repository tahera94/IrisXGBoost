from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import numpy as np

iris = load_iris()
print(iris.keys())

Data = iris["data"]
Target= iris["target"]
Target_names= iris["target_names"]
Feature_names= iris["feature_names"]
#print(Data)
#print(Target)
#print(Target_names)
#print(Feature_names)

# Iris data set contains 150 samples with 4 features each. The Goal is to classify them into 3 different species of Iris flowers.
#Splot the data into training and testing sets, XGBoost uses D matrics format for its data structure
X_train, X_test, y_train, y_test = train_test_split(
    Data, Target, test_size=0.2, random_state=42, stratify=Target
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train,y_train,test_size=0.2,random_state=42,stratify=y_train
)

#random state number does not matter here, just to ensure reproducibility that every time we run the code we get the same split.
#Stratify ensures that the class propotion in the train and test sets is similar to the original dataset.
# Create XGBoost classifier
params = {
    "objective": "multi:softprob",# for multi-class classification, predicts probabilities of each class
    "num_class": 3,
    "eval_metric": "mlogloss",# multi-class log loss, Penalized for wrong predictions
    "max_depth": 3,
    "eta": 0.1,# how much to update the model at each step, lower values make the model more robust to overfitting but require more trees
    "seed": 42
}
dtrain = xgb.DMatrix(X_train, label=y_train)
dval   = xgb.DMatrix(X_val, label=y_val)
dtest  = xgb.DMatrix(X_test)
evals = [(dtrain, "train"), (dval, "val")]
model = xgb.train(
    params,
    dtrain,
    num_boost_round=300,
    evals=evals,
    early_stopping_rounds=10,
    verbose_eval=True
)

# Make predictions
y_pred_proba = model.predict(dtest)
y_pred = np.argmax(y_pred_proba, axis=1)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris["target_names"]))