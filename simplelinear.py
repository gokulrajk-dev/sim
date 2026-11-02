import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

data = pd.read_csv('SIMPLE LINEAR REGRESSION/Boston.csv')
print(data.head())
X_simple = data[['rm']]  
y = data['medv']  
X_train, X_test, y_train, y_test = train_test_split(X_simple, y, test_size=0.2, random_state=42)
model_simple = LinearRegression()
model_simple.fit(X_train, y_train)
y_pred_simple = model_simple.predict(X_test)
print("Simple Linear Regression - MSE:", mean_squared_error(y_test, y_pred_simple))
print("Simple Linear Regression - R^2:", r2_score(y_test, y_pred_simple))
plt.scatter(X_test, y_test, color='blue', label='Actual Prices')
plt.scatter(X_test, y_pred_simple, color='red', label='Predicted Prices')
plt.title('Simple Linear Regression: RM vs PRICE')
plt.xlabel('Average Number of Rooms (RM)')
plt.ylabel('House Price (PRICE)')
plt.legend()
plt.show()
