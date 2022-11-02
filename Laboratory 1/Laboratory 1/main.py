import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

#1 
housingdata = pd.read_csv('HousingData.csv')
observations = housingdata.head(20)

print("\nFirst 20 rows:")
print (observations)
print("\n")
print(housingdata.isnull().sum())

#3
print("\nMedian:")
print(housingdata.median())

min_data = housingdata.min()
print("\nMin:")
print(min_data)

max_data = housingdata.max()
print("\nMax:")
print(max_data)

print("\nRange:")
print(max_data - min_data)

print("\nSkew:")
print(housingdata.skew())

check_crim_data = housingdata.loc[housingdata['CRIM'] > 30.0]
print("\nCheck CRIM column for outliers:")
print(check_crim_data.sort_values(by=['CRIM']))
rows_count = check_crim_data.count()[0]
print("\nNo. of rows: " + str(rows_count))

# #4
print("\nData Types:")
print(housingdata.dtypes)

print("\nChanged Types:")

changed_dtypes = {'TAX': float}
housingdata = housingdata.astype(changed_dtypes)
print(housingdata.dtypes)

print("\nModified dataset:")
observations = housingdata.head(30)
print(observations)

#5
print("\nNaN values:")
print(housingdata.isnull().sum())

housingdata = housingdata.fillna(housingdata.median())

changed_dtypes = {'CHAS': bool}
housingdata = housingdata.astype(changed_dtypes)
print(housingdata.dtypes)

get_rows = housingdata.head(20)
print("\nDatased without NaN values:")
print(get_rows)

print("\nNew check for NaN values:")
print(housingdata.isnull().sum())

#6
for x in housingdata.columns:
    if (x == 'CHAS'):
        continue
    plt.hist(housingdata[x], color = 'gray', edgecolor = 'black',
         bins = int(35000/1000))
    plt.axvline(housingdata[x].mean(), color='yellow', linestyle='dashed', linewidth=1)
    plt.axvline(housingdata[x].median(), color='green', linestyle='dashed', linewidth=1)
    plt.title(x)
    plt.show()

#7
river_housingdata = housingdata.loc[housingdata['CHAS'] == 1]
non_river_housingdata = housingdata.loc[housingdata['CHAS'] == 0]

for x in river_housingdata.columns:
    if (x == 'CHAS'):
        continue
    plt.boxplot(river_housingdata[x])
    plt.title("NEAR RIVER | " + x)
    plt.show()

for x in non_river_housingdata.columns:
    if (x == 'CHAS'):
        continue
    plt.boxplot(non_river_housingdata[x])
    plt.title("NOT NEAR THE RIVER | " + x)
    plt.show()

#8
plt.scatter(housingdata['CRIM'], housingdata['CHAS'])
plt.title("CRIM and CHAS")
plt.show()

plt.scatter(housingdata['CRIM'], housingdata['RM'])
plt.title("CRIM and RM")
plt.show()

plt.scatter(housingdata['CRIM'], housingdata['AGE'])
plt.title("CRIM and AGE")
plt.show()

plt.scatter(housingdata['CRIM'], housingdata['DIS'])
plt.title("CRIM and DIS")
plt.show()

plt.scatter(housingdata['CRIM'], housingdata['TAX'])
plt.title("CRIM and TAX")
plt.show()

plt.scatter(housingdata['CRIM'], housingdata['LSTAT'])
plt.title("CRIM and LSTAT")
plt.show()

plt.scatter(housingdata['CRIM'], housingdata['MEDV'])
plt.title("CRIM and MEDV")
plt.show()

plt.scatter(housingdata['CHAS'], housingdata['RM'])
plt.title("CHAS and RM")
plt.show()

plt.scatter(housingdata['CHAS'], housingdata['AGE'])
plt.title("CHAS and AGE")
plt.show()

plt.scatter(housingdata['CHAS'], housingdata['DIS'])
plt.title("CHAS and DIS")
plt.show()

plt.scatter(housingdata['CHAS'], housingdata['TAX'])
plt.title("CHAS and TAX")
plt.show()

plt.scatter(housingdata['CHAS'], housingdata['LSTAT'])
plt.title("CHAS and LSTAT")
plt.show()

plt.scatter(housingdata['CHAS'], housingdata['MEDV'])
plt.title("CHAS and LSTAT")
plt.show()

plt.scatter(housingdata['RM'], housingdata['AGE'])
plt.title("RM and AGE")
plt.show()

plt.scatter(housingdata['RM'], housingdata['DIS'])
plt.title("RM and DIS")
plt.show()

plt.scatter(housingdata['RM'], housingdata['LSTAT'])
plt.title("RM and LSTAT")
plt.show()

plt.scatter(housingdata['RM'], housingdata['MEDV'])
plt.title("RM and MEDV")
plt.show()

plt.scatter(housingdata['AGE'], housingdata['DIS'])
plt.title("AGE and DIS")
plt.show()

plt.scatter(housingdata['AGE'], housingdata['TAX'])
plt.title("AGE and TAX")
plt.show()

plt.scatter(housingdata['AGE'], housingdata['LSTAT'])
plt.title("AGE and LSTAT")
plt.show()

plt.scatter(housingdata['AGE'], housingdata['MEDV'])
plt.title("AGE and MEDV")
plt.show()

plt.scatter(housingdata['DIS'], housingdata['TAX'])
plt.title("DIS and TAX")
plt.show()

plt.scatter(housingdata['DIS'], housingdata['LSTAT'])
plt.title("DIS and LSTAT")
plt.show()

plt.scatter(housingdata['DIS'], housingdata['MEDV'])
plt.title("DIS and MEDV")
plt.show()

plt.scatter(housingdata['TAX'], housingdata['LSTAT'])
plt.title("TAX and LSTAT")
plt.show()

plt.scatter(housingdata['TAX'], housingdata['MEDV'])
plt.title("TAX and MEDV")
plt.show()

plt.scatter(housingdata['LSTAT'], housingdata['MEDV'])
plt.title("LSTAT and MEDV")
plt.show()

#9
corr = housingdata.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)
plt.show()

#10
X = housingdata[['LSTAT','RM']]
y = housingdata[['MEDV']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
linreg = LinearRegression()
linreg = linreg.fit(X_train, y_train)
y_pred = linreg.predict(X_test)

lin_model = pd.DataFrame(y_pred, columns=['Predicted_MEDV'])
lin_model['Actual_MEDV'] = y_test.to_numpy()
print(lin_model.head(20))

print('MSE:', mean_squared_error(y_test, y_pred, squared=True))
print('RMSE:', mean_squared_error(y_test, y_pred, squared=False))
print('R2:', r2_score(y_test, y_pred))