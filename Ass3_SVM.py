import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn import svm, preprocessing
from statistics import mean
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#load the dataset from github
df = pd.read_csv(
    "https://raw.githubusercontent.com/haixiaodai/public/0f8089d966b6c91226f0afcfd02b116ae59d91a0/Final%20Data.csv")
print("Any missing sample in dataset:", df.isnull().values.any())
df = df.drop(columns=['TotalComp', 'bonus', 'basePay'])

# transform income level using label encoder
le = preprocessing.LabelEncoder()
Y_train_label = df.IncomeGroup.values.astype(object)
le.fit(Y_train_label)
Y_encoded = le.transform(Y_train_label)

# Split X set.
X = df.iloc[:, :-1]
# transform other dataset using one hot encoder.
X = pd.get_dummies(X)
# Split train and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, Y_encoded, test_size=0.3, random_state=30)

# Create a scaler to snandardise features
usescaler = input("Use scaler? Press y for yes other for no: ")
if usescaler == "y":
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

print("Programme will take 5-10 minutes to run, please wait......")
# Create grid to find best parameters.
params_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                'C': [1, 10, 100, 1000]},
               {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

svm_model = GridSearchCV(svm.SVC(), params_grid)
svm_model.fit(X_train, y_train)
print('Best score for training data:', svm_model.best_score_, "\n")

# View the best parameters for the model found using grid search
print('Best C:', svm_model.best_estimator_.C, "\n")
print('Best Kernel:', svm_model.best_estimator_.kernel, "\n")
print('Best Gamma:', svm_model.best_estimator_.gamma, "\n")
# Build the final model based on best parameters.
final_model = svm_model.best_estimator_
y_pred=final_model.predict(X_test)
scoring = ['accuracy', 'recall_macro', 'precision_macro', 'f1_macro'] 
# Cross Validation.
scores = cross_validate(final_model, X, Y_encoded, cv=10, scoring=scoring)
print(sorted(scores.keys()))

# accuracy
print("Mean accuracy: %.3f%%" % (mean(scores['test_accuracy']) * 100))

# precision
print("Mean precision: %.3f " % (mean(scores['test_precision_macro'])))

# recall
print("Mean recall: %.3f" % (mean(scores['test_recall_macro'])))

# # F1 (F-Measure)
print("F1: %.3f" % (mean(scores['test_f1_macro'])))

# Calculate accuracy for the model.
accuracy = final_model.score(X_test, y_test)
accuracy_scaled = final_model.score(X_test, y_test)
print("Training set score for SVM: %f" % final_model.score(X_train, y_train))
print("Testing  set score for SVM: %f" % final_model.score(X_test, y_test))

# Generate Confusion Matrix.
cm = confusion_matrix(y_test, y_pred, labels=final_model.classes_)

# pretty print Confusion Matrix as a heatmap
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["low","average","high"])
disp.plot()
plt.show()

quit()
