import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
data = {
    'study_hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'marks': [35, 40, 50, 55, 65, 70, 75, 85, 90, 95]
}

df = pd.DataFrame(data)
df.to_csv("study_hours_marks.csv", index=False)

dataset = pd.read_csv("study_hours_marks.csv")

X = dataset[['study_hours']]   
y = dataset['marks']           

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)

m = model.coef_[0]
c = model.intercept_

print(f"Linear Regression Equation:")
print(f"Marks = {m:.2f} × Study Hours + {c:.2f}")

user_hours = float(input("Enter number of study hours: "))
user_df = pd.DataFrame({'study_hours': [user_hours]})

predicted_marks = model.predict(user_df)
print(f"Predicted Marks: {predicted_marks[0]:.2f}")
plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Study Hours vs Marks Prediction")
plt.show()
