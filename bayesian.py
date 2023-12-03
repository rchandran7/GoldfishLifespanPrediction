from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt

# Read training and testing sets
df_train = pd.read_csv('new_fish_train.csv')
df_test = pd.read_csv('new_fish_test.csv')

# Preprocess 'Gender' column for training set
le = LabelEncoder()
df_train['Gender'] = le.fit_transform(df_train['Gender'])

# Preprocess 'Gender' column for testing set
df_test['Gender'] = le.transform(df_test['Gender'])

# Split the data into features and target variable for training set
X_train = df_train[['average_length(inches))', 'average_weight(inches))', 'ph_of_water', 'Gender', 'idlewater', 'lakes', 'ponds', 'rivers', 'slowmovingwaters']]
y_train = df_train['life_span']

# Split the data into features and target variable for testing set
X_test = df_test[['average_length(inches))', 'average_weight(inches))', 'ph_of_water', 'Gender', 'idlewater', 'lakes', 'ponds', 'rivers', 'slowmovingwaters']]
y_test = df_test['life_span']

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the Bayesian Ridge Regression model
bayesian_ridge = BayesianRidge()

# Define the parameter grid to search
param_grid = {
    'alpha_1': [1e-6, 1e-5, 1e-4, 1e-3],
    'alpha_2': [1e-6, 1e-5, 1e-4, 1e-3],
    'lambda_1': [1e-6, 1e-5, 1e-4, 1e-3],
    'lambda_2': [1e-6, 1e-5, 1e-4, 1e-3],
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(bayesian_ridge, param_grid, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(X_train_scaled, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test_scaled)

# Print actual vs predicted values
results = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
print(results)

mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)

plt.scatter(y_test, predictions)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.show()
