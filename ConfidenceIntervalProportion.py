import pandas as pd
import statsmodels.stats.proportion as smp

df = pd.read_csv("https://raw.githubusercontent.com/haixiaodai/public/main/Final%20Data.csv")
# Filter and exact data
finaldf = df[(df.gender == "Female") & (df.edu == "College")]
# Count data
total = df.shape[0]
filtercount = finaldf.shape[0]
# Calculate confidence interval
lower, upper = smp.proportion_confint(filtercount, total, alpha=0.05, method="normal")
print("C.I. of proportion is %.3f and %.3f" % (lower, upper))
