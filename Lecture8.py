from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
X = iris.data
y = iris.target

X_train , X_test , y_train , y_test = train_test_split(X , y , random_state=4)
knn = KNeighborsClassifier(n_neighbors=5)
"""
n_neighbors=5
考慮數據點附近的 5 個 neighbor，將 5 個值相加取平均
"""

knn.fit(X_train , y_train)
y_pred = knn.predict(X_test)

print(knn.score(X_test , y_test))

from sklearn.model_selection import cross_val_score
KNN = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(KNN , X , y , cv=5 , scoring="accuracy")

print(scores)
print(scores.mean())

import matplotlib.pyplot as plt
k_range = range(1,31)
k_scores = []
for k in k_range:
    Knn = KNeighborsClassifier(n_neighbors=k)
    Scores = cross_val_score(Knn , X , y , cv=10 , scoring="accuracy")  # for Classification
    # loss = -cross_val_score(Knn , X , y , cv=10 , scoring="mean_square_error")  # for Regression
    """
    scoring="accuracy" for Classification
    scoring="mean_squared_error" for Regression
    """ 
    k_scores.append(Scores.mean())

plt.plot(k_range , k_scores)
plt.xlabel("Value of K for KNN")
plt.ylabel("Cross-Validated Accuracy")
plt.show()