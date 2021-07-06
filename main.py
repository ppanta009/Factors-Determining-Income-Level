# import pandas
import pandas as pd
import numpy as np
import seaborn as sns
import math
from scipy.stats import skew
from matplotlib import pyplot
from scipy.stats import norm
import scipy.stats as stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# read in a .csv file into Pandas DataFrame
df = pd.read_csv("https://raw.githubusercontent.com/haixiaodai/public/main/Final%20Data.csv", index_col=0)
print(df)

# Check if there is any missing value
print("Has missing value: ", df.isnull().values.any())

# get the descreptive statistics - skewness of the dataframe
print("The skewness of age:", skew(df.age.values))
print("The skewness of basepay:", skew(df.basePay.values))
print("The skewness of bonus:", skew(df.bonus.values))
print("The skewness of Total compensation: ", skew(df.TotalComp.values))
print(df.describe())

# get the descriptive statistics - variance of the dataframe
print("Variance: \n", df.var())


# Function to draw distribution diagram for continuous values
def drawdistribution(datalist, title):  # datalist = dataset and title is the title of chart
    datalist.sort()  # sort the data so that line of data point wont be shown in distribution
    mean = np.mean(datalist)  # calculate mean
    std = np.std(datalist)  # calculate standard deviation
    dist = norm(mean, std)  # creates the distribution
    probabilities = [dist.pdf(value) for value in
                     datalist]  # for each value in values, compute its probability of being drawn from the given normal distribution
    pyplot.plot(datalist, probabilities)  # values on the X-axis, probability scores on the Y-axis
    pyplot.title(str(title) + " Distribution")  # tile of the chart
    pyplot.xlabel(str(title) + " Value")  # title of X-axis
    pyplot.ylabel("Probability")  # title of Y-axis
    pyplot.show()


# Calling function for each continous variable in the dataset
drawdistribution(df.age.values, "Age")
drawdistribution(df.basePay.values, "Base Pay")
drawdistribution(df.bonus.values, "Bonus")
drawdistribution(df.TotalComp.values, "Total Compensation")


# function to draw histogram for continous varibales in the dataset
def drawhistogram(data, title):
    pyplot.hist(data, 10)
    pyplot.title(str(title) + " Distribution")  # tile of the chart
    pyplot.xlabel(str(title) + " Value")  # title of X-axis
    pyplot.ylabel("Probability")  # title of Y-axis
    pyplot.show()


# calling functions to draw the histogram of continuous varible
drawhistogram(df.age.values, "Age")
drawhistogram(df.basePay.values, "Base Pay")
drawhistogram(df.bonus.values, "Bonus")
drawhistogram(df.TotalComp.values, "Total Compensation")

# box plot base pay and total compensation to better see the distribution range
pyplot.figure(figsize=(15, 6))
pyplot.subplot(1, 2, 1)
sns.boxplot(y=df["basePay"], color="blue")
pyplot.subplot(1, 2, 2)
sns.boxplot(y=df["TotalComp"], color="yellow")
pyplot.show()

# box plot showing bonus and age
pyplot.figure(figsize=(15, 6))
pyplot.subplot(1, 2, 1)
sns.boxplot(y=df["bonus"], color="green")
pyplot.subplot(1, 2, 2)
sns.boxplot(y=df["age"], color="red")
pyplot.show()

# function to generate the distribution of categorical variable
'''this function takes following argument:
data = dataframe of the variable that chart will be generated
x = Label on X-axis
y = Label on Y-axis
z = title of chart'''


def barchar(data, x, y, z):
    value = data.value_counts()
    sns.set_style("whitegrid")
    pyplot.figure(figsize=(18, 8))
    sns.barplot(x=value.index, y=value.values)
    pyplot.title(z)
    pyplot.xlabel(x)
    pyplot.ylabel(y)
    pyplot.show()


barchar(df.gender, "Gender", "Number of People", "Gender Distribution")
barchar(df.jobTitle, "Title of Job", "Number of People", "Job Title Distribution")
barchar(df.perfEval, "Rating from 1 to 5", "Number of People", "Performance Evaluation Distribution")
barchar(df.edu, "Education Level", "Number of People", "Education Level Chart")
barchar(df.dept, "Department", "Number of People", "Distribution of Department")
barchar(df.seniority, "Rating from 1 to 5", "Number of People", "Seniority Distribution")

'''our assumption based on research paper is people under 45 are said to be experienced or mature
so lets divide the age between 2 group below 45 and above 45 and list our some diagram'''
# this part of code aims to look at distribution to one of our hypothesis

age_lessthan45 = df.age[(df.age <= 45)]
age_morethan45 = df.age[(df.age > 45)]

x = ["18-45", "45-65"]
y = [len(age_lessthan45.values), len(age_morethan45.values)]
pyplot.figure(figsize=(15, 6))
sns.barplot(x=x, y=y, palette="rocket")
pyplot.title("Number of experienced and unexperienced people")
pyplot.xlabel("Age")
pyplot.ylabel("Number of people")
pyplot.show()

'''compensation distribution to better view the data'''
tc30_60 = df.TotalComp[(df.TotalComp >= 30000) & (df.TotalComp <= 60000)]
tc60_90 = df.TotalComp[(df.TotalComp >= 60001) & (df.TotalComp <= 90000)]
tc90_120 = df.TotalComp[(df.TotalComp >= 90001) & (df.TotalComp <= 120000)]
tc120_150 = df.TotalComp[(df.TotalComp >= 120001) & (df.TotalComp <= 150000)]
tc_150over = df.TotalComp[df.TotalComp >= 150001]

tcx = ["$ 30,000 - 60,000", "$ 60,001 - 90,000", "$ 90,001 - 120,000 ", "$ 120,001 - 150,000", "$150,000 and Over"]
tcy = [len(tc30_60.values), len(tc60_90.values), len(tc90_120.values), len(tc120_150.values), len(tc_150over.values)]

pyplot.figure(figsize=(16, 6))
sns.barplot(x=tcx, y=tcy, palette="Set2")
pyplot.title("Total Compensation")
pyplot.xlabel("Compensation")
pyplot.ylabel("Number of Peopple")
pyplot.show()

'''Inferential Statistics - Confidence Interval of Mean and Population'''
def infStatMean(data, x):
    s_mean = data.mean()
    z_critical = stats.norm.ppf(q=0.975)
    p_std = data.std()
    std_err = p_std / math.sqrt(len(data))
    margin_of_error = z_critical * std_err
    ci_lower = s_mean - margin_of_error
    ci_upper = s_mean + margin_of_error
    print("Margin of Error is of %s is: %f " % (x, margin_of_error))
    print("Standard Error of %s is: %f " % (x, std_err))
    print("Confidence Interval of the mean for %s : %.2f to %.2f \n" % (x, ci_lower, ci_upper))


infStatMean(df.age, "age")
infStatMean(df.basePay, "basepay")
infStatMean(df.bonus, "bonus")
infStatMean(df.TotalComp, "Total Compensation")
