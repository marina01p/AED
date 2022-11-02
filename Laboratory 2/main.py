import pandas as pd

#1
dataset = pd.read_csv('Lab_2_data.csv')
observations = dataset.head(5)

print("\nFirst 20 rows:")
print (observations)