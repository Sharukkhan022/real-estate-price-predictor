import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('data/cleaned_redfin_housing.csv')

# print(df.head())

# 2. Separate Features (X) and Target (y)
X = df.drop(columns=['Price (USD)'])
y = df['Price (USD)']

# 3. Train-Test Split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Make Predictions on Test Data
y_pred = model.predict(X_test)

# 6. Evaluate Performance
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5

print("Model Evaluation Metrics:")
print(f"R-squared: {r2:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}") 

# 7. Save the trained model to a file
joblib.dump(model, 'models/linear_regression_model.pkl')
print("Model saved to 'models/linear_regression_model.pkl'")

