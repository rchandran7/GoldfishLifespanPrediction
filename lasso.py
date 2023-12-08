from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt

df_train = pd.read_csv('new_fish_train.csv')
df_test = pd.read_csv('new_fish_test.csv')

le = LabelEncoder()
df_train['Gender'] = le.fit_transform(df_train['Gender'])

# Preprocess 'Gender' column for testing set
df_test['Gender'] = le.transform(df_test['Gender'])


X_train = df_train.drop(['life_span'], axis=1)
X_test = df_test.drop(['life_span'], axis=1)
y_train = df_train['life_span']
y_test = df_test['life_span']


# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


lasso = Lasso()
param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1.0],
}
grid_search = GridSearchCV(
    lasso, param_grid, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(X_train_scaled, y_train)


best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test_scaled)
results = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
print(results)

mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)

plt.figure(figsize=(10, 6))
plt.scatter(df_test['life_span'], predictions,
            label='LASSO Regression', alpha=0.7)
plt.xlabel('Actual life_span')
plt.ylabel('Predicted life_span')
plt.title('Scatter Plot for LASSO Regression Model Predictions on Test Set')
plt.legend()
plt.show()
