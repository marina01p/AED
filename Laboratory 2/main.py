import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
from statsmodels.discrete.discrete_model import Probit
#1
dataset = pd.read_csv('Lab_2_data.csv')
observations = dataset.head(5)

print("\nFirst 5 rows:")
print (observations)

# column_names = list(dataset.columns.values)
# print(column_names)

dataset = dataset.drop(['Unnamed: 0.1', 'Unnamed: 0'], axis=1)

dataset_cols = ['lfp','wc','hc']
le = LabelEncoder()
dataset[dataset_cols] = dataset[dataset_cols].apply(le.fit_transform)

print("\nNew dataset | First 5 rows:")
observations = dataset.head(5)
print (observations)

#3
print("\nSummary statistics:")
dataset_summary = dataset.agg(
    {
        "lfp": ["min", "max", "median", "skew", "mean"],
        "k5": ["min", "max", "median", "mean", "skew"],
        "k618": ["min", "max", "median", "mean", "skew"],
        "age": ["min", "max", "median", "mean", "skew"],
        "wc": ["min", "max", "median", "mean", "skew"],
        "hc": ["min", "max", "median", "mean", "skew"],
        "lwg": ["min", "max", "median", "mean", "skew"],
        "inc": ["min", "max", "median", "mean", "skew"],
    }
)

print(dataset_summary)

dataset.drop(dataset[dataset['lwg'] <= 0].index, inplace = True)
dataset.drop(dataset[dataset['inc'] <= 0].index, inplace = True)

print("\nModified table | Summary statistics:")
dataset_summary = dataset.agg(
    {
        "lfp": ["min", "max", "median", "skew", "mean"],
        "k5": ["min", "max", "median", "mean", "skew"],
        "k618": ["min", "max", "median", "mean", "skew"],
        "age": ["min", "max", "median", "mean", "skew"],
        "wc": ["min", "max", "median", "mean", "skew"],
        "hc": ["min", "max", "median", "mean", "skew"],
        "lwg": ["min", "max", "median", "mean", "skew"],
        "inc": ["min", "max", "median", "mean", "skew"],
    }
)

print(dataset_summary)

#4
print("\nNaN values:")
print(dataset.isnull().sum())

#5
# for x in dataset.columns:
#     if (x == 'wc' or x == 'hc'):
#         continue
#     plt.hist(dataset[x], color = 'gray', edgecolor = 'black',
#          bins = int(35000/1000))
#     plt.axvline(dataset[x].mean(), color='yellow', linestyle='dashed', linewidth=1)
#     plt.axvline(dataset[x].median(), color='green', linestyle='dashed', linewidth=1)
#     plt.title("Histogram for " + x + " column")
#     plt.show()

#6
# corr = dataset.corr()
# f, ax = plt.subplots(figsize=(7, 5))
# cmap = sns.diverging_palette(230, 20, as_cmap=True)
# sns.heatmap(corr, annot=True, cmap=cmap)
# plt.show()

#7
print("\nProbit model:")
X = dataset[['k5', 'k618', 'age', 'wc', 'hc', 'lwg', 'inc']]
Y = dataset['lfp']
X = sm.add_constant(X)
model = Probit(Y, X.astype(float))
probit_model = model.fit()
print(probit_model.summary())