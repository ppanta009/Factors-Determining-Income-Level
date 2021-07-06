# evaluate model on the raw dataset
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import pandas as pd

# load the dataset
df = pd.read_csv("https://raw.githubusercontent.com/haixiaodai/public/main/Final%20Data.csv", index_col=0)

# retrieve DataFrame's content as a matrix
# the .values() function serves the same purpose as .to_numpy()
data = df._get_numeric_data().values

# split into input and output elements
X, y = data[:, :-1], data[:, -1]

# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

# fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# evaluate the model
y_hat = model.predict(X_test)

# evaluate predictions
mae = mean_absolute_error(y_test, y_hat)
print('MAE: %.3f' % mae)
