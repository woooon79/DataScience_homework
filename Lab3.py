import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.preprocessing as preprocessing
from sklearn import linear_model

#Set print all values (do not skip)
pd.set_option('display.max_row', 300)
'''
# read csv file
df = pd.read_csv('C:/Users/Howoon/Desktop/LAB3/bmi_data_lab3.csv')
# print statistical data
print(df.describe())
# print feature names & data types
print(df.info())

# decompose the dataframe by BMI
e_weak = df[df['BMI'] == 0]
weak = df[df['BMI'] == 1]
normal = df[df['BMI'] == 2]
overweight = df[df['BMI'] == 3]
obesity = df[df['BMI'] == 4]


# define the function drawing histogram
def draw_hist(row, bmi, title):
    ax[row, 0].hist(bmi['Height (Inches)'], bins=10)
    ax[row, 0].set_title(title)
    ax[row, 0].set_xlabel("height")
    ax[row, 0].set_ylabel("frequency")

    ax[row, 1].hist(bmi['Weight (Pounds)'], bins=10)
    ax[row, 1].set_title(title)
    ax[row, 1].set_xlabel("weight")
    ax[row, 1].set_ylabel("frequency")


# Set subplots
fig, ax = plt.subplots(5, 2, figsize=(15, 24))

draw_hist(0, e_weak, 'Extremely weak')
draw_hist(1, weak, 'Weak')
draw_hist(2, normal, 'Normal')
draw_hist(3, overweight, 'Overweight')
draw_hist(4, obesity, 'Obesity')

plt.show()

# transform Series to Dataframe
height = pd.DataFrame(df['Height (Inches)'])
weight = pd.DataFrame(df['Weight (Pounds)'])

# Standard Scaler
scaler = preprocessing.StandardScaler()
# fit and transform
scaled_height = scaler.fit_transform(height)
scaled_weight = scaler.fit_transform(weight)
# transform array to dataframe
scaled_height = pd.DataFrame(scaled_height, columns=['height'])
scaled_weight = pd.DataFrame(scaled_weight, columns=['weight'])
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6, 5))
ax1.set_title('Height')
sns.kdeplot(scaled_height['height'], ax=ax1)
ax2.set_title('Weight')
sns.kdeplot(scaled_weight['weight'], ax=ax2)
plt.show()

# MinMax Scaler
scaler = preprocessing.MinMaxScaler()
# fit and transform
scaled_height = scaler.fit_transform(height)
scaled_weight = scaler.fit_transform(weight)
# transform array to dataframe
scaled_height = pd.DataFrame(scaled_height, columns=['height'])
scaled_weight = pd.DataFrame(scaled_weight, columns=['weight'])
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6, 5))
ax1.set_title('Height')
sns.kdeplot(scaled_height['height'], ax=ax1)
ax2.set_title('Weight')
sns.kdeplot(scaled_weight['weight'], ax=ax2)
plt.show()

# Robust Scaler
scaler = preprocessing.RobustScaler()
# fit and transform
scaled_height = scaler.fit_transform(height)
scaled_weight = scaler.fit_transform(weight)
# transform array to dataframe
scaled_height = pd.DataFrame(scaled_height, columns=['height'])
scaled_weight = pd.DataFrame(scaled_weight, columns=['weight'])
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6, 5))
ax1.set_title('Height')
sns.kdeplot(scaled_height['height'], ax=ax1)
ax2.set_title('Weight')
sns.kdeplot(scaled_weight['weight'], ax=ax2)
plt.show()



'''
# erase the wrong value and reread bmi_data_lab3
df = pd.read_csv('C:/Users/Howoon/Desktop/LAB3/bmi_data_lab3.csv')

# print the number of rows with NaN
sum=0
for i in range(len(df)):
    if df.T.isna().sum()[i]>0:
        sum=sum+1
print('number of rows with NaN:',sum)
# print the number of missing value for each columns
print(df.isna().sum())

# print the all rows without NAN
print(df.dropna(axis=0, how='any'))

# Fill NAN with using ffill methods
df.fillna(axis=0, method='ffill', inplace=True)
print(df)

# erase the wrong value and reread bmi_data_lab3
df_origin = pd.read_csv('C:/Users/Howoon/Desktop/LAB3/bmi_data_lab3.csv')

# select Height and Weight columns
df = df_origin.loc[:, ['Height (Inches)', 'Weight (Pounds)']]

# select weight values which have NaN height value
nan_weight = df[df['Height (Inches)'].isna()].loc[:, 'Weight (Pounds)']
# select height values which have NaN weight value
nan_height = df[df['Weight (Pounds)'].isna()].loc[:, 'Height (Inches)']

# drop all rows which has NaN value
df.dropna(axis=0, how='any', inplace=True)
# change type to numpy array
height = df['Height (Inches)'].to_numpy()
weight = df['Weight (Pounds)'].to_numpy()

# set linear regression
E = linear_model.LinearRegression()
E.fit(height[:, np.newaxis], weight)
px = np.array([height.min() - 1, height.max() + 1])
py = E.predict(px[:, np.newaxis])

# scatter plot (which has both height and weight values)
plt.scatter(height, weight)
# scatter plot (which has only height values)
# predict weight values using linear regression
plt.scatter(nan_height, E.predict(nan_height[:, np.newaxis]), c='red')

# scatter plot (which has only weight values)
# predict height values using line
predict_height = (nan_weight - E.intercept_) / E.coef_
plt.scatter(predict_height, nan_weight, c='red')

# draw line(linear regression)
plt.plot(px, py, color='black')
plt.xlabel('Height (Inches)')
plt.ylabel('Weight (Pounds)')
plt.show()


def group_LinearRegression(col, value):
    # select Height and Weight columns in female groups
    df = df_origin[df_origin[col] == value].loc[:, ['Height (Inches)', 'Weight (Pounds)']]

    # select weight values which have NaN height value
    nan_weight = df[df['Height (Inches)'].isna()].loc[:, 'Weight (Pounds)']
    # select height values which have NaN weight value
    nan_height = df[df['Weight (Pounds)'].isna()].loc[:, 'Height (Inches)']

    # drop all rows which has NaN value
    df.dropna(axis=0, how='any', inplace=True)
    # change type to numpy array
    height = df['Height (Inches)'].to_numpy()
    weight = df['Weight (Pounds)'].to_numpy()

    # set linear regression
    E = linear_model.LinearRegression()
    E.fit(height[:, np.newaxis], weight)
    px = np.array([height.min() - 1, height.max() + 1])
    py = E.predict(px[:, np.newaxis])

    # scatter plot (which has both height and weight values)
    plt.scatter(height, weight)
    # scatter plot (which has only height values)
    # predict weight values using linear regression
    if len(nan_height) > 0:
        plt.scatter(nan_height, E.predict(nan_height[:, np.newaxis]), c='red')

    # scatter plot (which has only weight values)
    # predict height values using line
    predict_height = (nan_weight - E.intercept_) / E.coef_
    if len(nan_weight) > 0:
        plt.scatter(predict_height, nan_weight, c='red')

    # draw line(linear regression)
    plt.plot(px, py, color='black')
    plt.xlabel('Height (Inches)')
    plt.ylabel('Weight (Pounds)')
    plt.title(value)
    plt.show()


group_LinearRegression('Sex', 'Female')
group_LinearRegression('Sex', 'Male')

# Linear Regression for each BMI
# __There is no BMI 0 and 4 values in file
# group_LinearRegression('BMI',0)
group_LinearRegression('BMI', 1)
group_LinearRegression('BMI', 2)
group_LinearRegression('BMI', 3)
# group_LinearRegression('BMI',4)
















