import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn import preprocessing
from statistics import mean
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

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
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
gnb = GaussianNB()
param_grid = {'var_smoothing': [1e-8, 1e-9, 1e-10]}
gaussian_CV = GridSearchCV(estimator=gnb, param_grid=param_grid, cv=10, verbose=False)
gaussian_CV.fit(X_train, y_train)
print(gaussian_CV.best_params_)
y_pred = gaussian_CV.predict(X_test)
print(gaussian_CV.score(X_train, y_train))
scoring = ['accuracy', 'recall_macro', 'precision_macro', 'f1_macro']
scores = cross_validate(gaussian_CV, X, Y_encoded, cv=10, scoring=scoring)
print(sorted(scores.keys()))
# accuracy
print("Mean accuracy: %.3f%%" % (mean(scores['test_accuracy']) * 100))

# precision
print("Mean precision: %.3f " % (mean(scores['test_precision_macro'])))

# recall
print("Mean recall: %.3f" % (mean(scores['test_recall_macro'])))

# # F1 (F-Measure)
print("F1: %.3f" % (mean(scores['test_f1_macro'])))

print("Training set score for SVM: %f" % gaussian_CV.score(X_train, y_train))
print("Testing  set score for SVM: %f" % gaussian_CV.score(X_test, y_test))

cm = confusion_matrix(y_test, y_pred, labels=gaussian_CV.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["low","average","high"])
disp.plot()
plt.show()
quit()