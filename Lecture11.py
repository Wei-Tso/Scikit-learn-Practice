from sklearn import svm
from sklearn import datasets

clf = svm.SVC()
iris = datasets.load_iris()
X , y = iris.data , iris.target
clf.fit(X , y)

# method 1 : pickle
import pickle
with open("clf.pickle" , "wb") as f:
    pickle.dump(clf , f)

'''
讀取以儲存的檔案
with open("clf.pickle" , "rb") as f:
    clf2 = pickle.load(f)
    print(clf2.predict(X[0:1]))
'''

# method 2 : joblib
from sklearn.externals import joblib
joblib.dump(clf , "clf2.pkl")

'''
讀取以儲存的檔案
clf3 = joblib.load("clf2.pkl")
print(clf3.predict(X[0:1]))
'''