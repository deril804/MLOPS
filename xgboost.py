import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load data
iris = load_iris()
X = iris.data
y = iris.target

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define parameters for XGBoost
params = {
    'objective': 'multi:softmax',
    'num_class': 3,
    'eval_metric': 'mlogloss',
}

# Create DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Start MLflow run
with mlflow.start_run():
    # Train XGBoost model
    model = xgb.train(params, dtrain, evals=[(dtest, 'eval')])

    # Log model and parameters to MLflow
    mlflow.xgboost.log_model(model, "xgboost_model")
    mlflow.log_params(params)
