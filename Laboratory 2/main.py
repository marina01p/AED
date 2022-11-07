import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
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

dataset.drop(dataset[dataset['lwg'] < 0].index, inplace = True)
dataset.drop(dataset[dataset['inc'] < 0].index, inplace = True)

print("\nModified table | Summary statistics:")
new_dataset_summary = dataset.agg(
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

print(new_dataset_summary)

#4

