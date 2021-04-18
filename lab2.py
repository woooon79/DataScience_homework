
import pandas as pd
import numpy as np

arr=np.array([3.,'?',2.,5.,'*',4.,5.,6.,'+',3.,2.,'&',5.,'?',7.,'!'])
arr=arr.reshape(4,4)

df=pd.DataFrame(arr)
print(df)

df.replace({"?":np.nan,"!":np.nan,"+":np.nan,"*":np.nan,"&":np.nan},inplace=True)
df=df.apply(pd.to_numeric)
print(df)

print(df.isna().any())
print(df.isna().sum())

print(df.dropna(axis=0,how="any"))
print(df.dropna(axis=0,how="all"))
print(df.dropna(axis=0,thresh=1))
print(df.dropna(axis=0,thresh=2))


mean=df.mean()
median=df.median()
print(df.fillna(100))
print(df.fillna(mean))
print(df.fillna(median))

print(df.fillna(axis=0,method='ffill'))
print(df.fillna(axis=0,method='bfill'))
