import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn import svm, preprocessing
from statistics import mean
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# load the dataset from github
df = pd.read_csv(
    "https://raw.githubusercontent.com/haixiaodai/public/0f8089d966b6c91226f0afcfd02b116ae59d91a0/Final%20Data.csv")
# df = pd.read_csv(
#     "/Users/hax/Downloads/Final Data_High.csv")

print("Any missing sample in dataset:", df.isnull().values.any())
df = df.drop(columns=['TotalComp', 'bonus', 'basePay'])

# transform income level using label encoder
le = preprocessing.LabelEncoder()
Y_train_label = df.IncomeGroup.values.astype(object)
le.fit(Y_train_label)
Y_encoded = le.transform(Y_train_label)
# Y_encoded = df.IncomeGroup.values


# Get X set
X = df.iloc[:, :-1]
# transform other dataset using onehotencoder
X = pd.get_dummies(X)
# Split train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y_encoded, test_size=0.3, random_state=30)
# Create a scaler to snandardise features
scaler = preprocessing.StandardScaler()
X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.transform(X_test)

print("Programme will take 5-10 minutes to run, please wait......")
# Create grid to find best parameters
params_grid = [{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
# Create linear SVM model
svm_model = GridSearchCV(svm.SVC(), params_grid)
svm_model.fit(X_train_scale, y_train)
print('Best C:', svm_model.best_estimator_.C, "\n")
# Build the final model based on best parameters.
finalmodel=svm_model.best_estimator_
y_pred = finalmodel.predict(X_test_scale)
scoring = ['accuracy', 'recall_macro', 'precision_macro', 'f1_macro']
scores = cross_validate(finalmodel, X, Y_encoded, cv=10, scoring=scoring)
print(sorted(scores.keys()))

# accuracy
print("Mean accuracy: %.3f%%" % (mean(scores['test_accuracy']) * 100))

# precision
print("Mean precision: %.3f " % (mean(scores['test_precision_macro'])))

# recall
print("Mean recall: %.3f" % (mean(scores['test_recall_macro'])))

# # F1 (F-Measure)
print("F1: %.3f" % (mean(scores['test_f1_macro'])))

# Calculate accuracy of the model
accuracy = finalmodel.score(X_test, y_test)
accuracy_scaled = finalmodel.score(X_test, y_test)
print("Training set score for SVM: %f" % finalmodel.score(X_train, y_train))
print("Testing  set score for SVM: %f" % finalmodel.score(X_test, y_test))

# Extract header of each column
headers = list(X.columns)
# Calculate coefficient of each feature in the dataset
f_weights = finalmodel.coef_[0].tolist()
f_weights_abs = [abs(x) for x in f_weights]
srt_list = list(zip(f_weights_abs, headers))
srt_list.sort(reverse=True)
for e in srt_list:
    print(e[1], ": ", e[0])

# Draw confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=finalmodel.classes_)

# pretty print Confusion Matrix as a heatmap
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["low", "average", "high"])
disp.plot()
plt.show()

quit()
