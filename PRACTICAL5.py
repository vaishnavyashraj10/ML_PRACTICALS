# Practical 5: Linear Regression

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Study hours and marks
X = [[1], [2], [3], [4], [5]]
y = [20, 30, 40, 50, 60]

# Create and train model
model = LinearRegression()
model.fit(X, y)

# Predict marks
predictions = model.predict(X)

# Evaluate model
mse = mean_squared_error(y, predictions)
r2 = r2_score(y, predictions)

print("Actual Marks:", y)
print("Predicted Marks:", predictions)
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("Mean Squared Error:", round(mse, 2))
print("R2 Score:", round(r2, 2))

# Predict marks for 6 study hours
print("Predicted marks for 6 hours:", model.predict([[6]])[0])