import tensorflow as tf
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.stats import zscore

# Load your data
df_train = pd.read_csv('new_fish_train.csv')
df_test = pd.read_csv('new_fish_test.csv')

# Label encoding for 'Gender'
le = LabelEncoder()
df_train['Gender'] = le.fit_transform(df_train['Gender'])
df_test['Gender'] = le.transform(df_test['Gender'])

# Extract features and target variable
X_train = df_train.drop(['life_span'], axis=1)
X_test = df_test.drop(['life_span'], axis=1)
y_train = df_train['life_span']
y_test = df_test['life_span']


model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))


# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=20, validation_split=0.2, verbose=0)

# Evaluate the model on the test set
predictions = model.predict(X_test).flatten()

results = pd.DataFrame({'Actual': df_test['life_span'], 'Expected': predictions})
print(results)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, label='Neural Network Predictions', alpha=0.7)
plt.xlabel('Actual life_span')
plt.ylabel('Predicted life_span')
plt.title('Scatter Plot for Neural Network Model Predictions on Test Set')
plt.legend()
plt.show()
