from sklearn.linear_model import Lasso  # Lasso regression model
# Metric for evaluating model performance
from sklearn.metrics import mean_squared_error
# Data preprocessing tools
from sklearn.preprocessing import StandardScaler, LabelEncoder
# Hyperparameter tuning using grid search
from sklearn.model_selection import GridSearchCV
import pandas as pd  # Data manipulation
import matplotlib.pyplot as plt  # Data visualization

# Read training and testing datasets
df_train = pd.read_csv('new_fish_train.csv')
df_test = pd.read_csv('new_fish_test.csv')

# Encode 'Gender' column using LabelEncoder
le = LabelEncoder()
df_train['Gender'] = le.fit_transform(df_train['Gender'])

# Preprocess 'Gender' column for testing set
df_test['Gender'] = le.transform(df_test['Gender'])

# Prepare feature and target variables for training and testing
X_train = df_train.drop(['life_span'], axis=1)
X_test = df_test.drop(['life_span'], axis=1)
y_train = df_train['life_span']
y_test = df_test['life_span']

# Feature scaling using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize Lasso regression model
lasso = Lasso()
param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1.0],
}

# Define hyperparameter grid for GridSearchCV
grid_search = GridSearchCV(
    lasso, param_grid, scoring='neg_mean_squared_error', cv=5)

# Perform Grid Search with cross-validation for hyperparameter tuning
grid_search.fit(X_train_scaled, y_train)
# Get the best hyperparameters from the grid search
best_params = grid_search.best_params_
# Retrieve the best model from the grid search
best_model = grid_search.best_estimator_
# Make predictions on the test set using the best model
predictions = best_model.predict(X_test_scaled)

# Create a DataFrame to compare actual and predicted values
results = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
print(results)

# Calculate Mean Squared Error (MSE) as a performance metric
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)

# Visualize the predictions using a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, alpha=0.7)
plt.xlabel('Actual life_span')
plt.ylabel('Predicted life_span')
plt.title('Scatter Plot for Lasso Regression Model Predictions on Test Set')
plt.show()
