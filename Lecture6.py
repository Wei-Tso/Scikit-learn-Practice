from sklearn import datasets
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

loaded_data = datasets.load_boston()
data_X = loaded_data.data
data_y = loaded_data.target

model = LinearRegression()
model.fit(data_X , data_y)

# y = ax + b
# 輸出 x 項的係數
print(model.coef_)
# 輸出 b
print(model.intercept_)

print(model.get_params())

# Linear Regression 是使用 R2 score
print(model.score(data_X , data_y))