import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn import tree
from sklearn import preprocessing


df = pd.read_csv(
    "https://raw.githubusercontent.com/haixiaodai/public/0f8089d966b6c91226f0afcfd02b116ae59d91a0/Final%20Data.csv")
print("Any missing sample in dataset:", df.isnull().values.any())
df = df.drop(columns=['TotalComp', 'bonus', 'basePay'])
le = preprocessing.LabelEncoder()
Y_train_label = df.IncomeGroup.values.astype(object)
le.fit(Y_train_label)
Y_encoded = le.transform(Y_train_label)
X = df.iloc[:, :-1]
X = pd.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(X, Y_encoded, test_size=0.3, random_state=30)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
print(clf.predict(y_test))