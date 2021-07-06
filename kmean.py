import pandas as pd
import numpy as np
from matplotlib import pyplot
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# read in a .csv file into Pandas DataFrame
df = pd.read_csv("https://raw.githubusercontent.com/haixiaodai/public/main/Final%20Data.csv", index_col=0)


'''taking only numerical value i.e. age and pay to determine the cluster so created new dataframe'''
data1 = df.iloc[:, 7:11]  # dataframe first containing details of compnesation only
data2 = df.iloc[:, 2:3]  # data frame 2 containing dage only
data = pd.concat([data2, data1], axis=1)  # appending two dataset and merging it into one

'''determining the number of clusters via elbow method'''
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans = kmeans.fit(data)
    wcss.append(kmeans.inertia_)

pyplot.figure(figsize=(12, 6))
pyplot.plot(range(1, 11), wcss, "bx-", linewidth=2, color="red")
pyplot.xlabel("K Value")
pyplot.xticks(np.arange(1, 11, 1))
pyplot.ylabel("WCSS")
pyplot.show()

'''Now we have determined the optimal number of cluster is either 4 or 5 so lets build scatterplot using standarization'''

# build Kmeans model
param = 4
kmeans = KMeans(n_clusters= param)
newdata = data.values   #convert into array for the purpose of visualization to aviod type error
print(newdata)

# apply standardization to determine the effeciency of the data.
scaler = StandardScaler()
kmeans = kmeans.fit(scaler.fit_transform(newdata))
centriods = kmeans.cluster_centers_
labels = kmeans.labels_

# reverse the centriod value to their original unit values
centriods_inv = scaler.inverse_transform(centriods)

#visualize the scatterplot to see the cluster
pyplot.figure(figsize=(12,8))
pyplot.scatter(newdata[:,1], newdata[:,0],c = labels.astype(float))
pyplot.scatter(centriods_inv[:,1], centriods_inv[:,0], marker = "x", c = "red")
pyplot.show()