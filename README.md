# EXP-10 Data Science Process on Complex Dataset
# AIM
To Perform Data Science Process on a complex dataset and save the data to a file.

# ALGORITHM
# Step 1
Read the given Data

# Step 2
Clean the Data Set using Data Cleaning Process

# Step 3
Apply Feature Generation/Feature Selection Techniques on the data set

# Step 4
Apply EDA /Data visualization techniques to all the features of the data set

# CODE
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

df = sns.load_dataset("tips")

df.head()

df.isnull().sum()

plt.figure(figsize=(5,5))

plt.title("Data with Outliers")

df.boxplot()

plt.show()

plt.figure(figsize=(5,5))

cols = ['size','tip','total_bill']

Q1 = df[cols].quantile(0.25)

Q3 = df[cols].quantile(0.75)

IQR = Q3 - Q1

df = df[~((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

plt.title("Dataset after removing outliers")

df.boxplot()

plt.show()

df['sex'].unique()

!pip install --upgrade category_encoders

from category_encoders import BinaryEncoder

be = BinaryEncoder()

data = be.fit_transform(df['sex'])

df = pd.concat([df,data],axis=1)

df

df['smoker'].unique()

data = be.fit_transform(df['smoker'])

df = pd.concat([df,data],axis=1)

df

df['day'].unique()

from sklearn.preprocessing import LabelEncoder,OrdinalEncoder

clim = ['Thur','Fri','Sat','Sun']

en= OrdinalEncoder(categories = [clim])

df['day']=en.fit_transform(df[["day"]])

df

df['time'].unique()

le = LabelEncoder()

df['time'] = le.fit_transform(df[["time"]])

df

df.drop('sex',axis=1,inplace=True)

df.drop('smoker',axis=1,inplace=True)

from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(df)

print("Min-max scaled data:")

print(scaled_data)

scaler = StandardScaler()

scaled_data = scaler.fit_transform(df)

print("Standard scaled data:")

print(scaled_data)

import seaborn as sns

sns.scatterplot(data=df)

sns.displot(df['size'],kde=True)

sns.scatterplot(x="total_bill", y="tip", data=df)

plt.title("Correlation between Tip Amount and Total Bill Amount")

plt.show()

df["tip_percent"] = df["tip"] / df["total_bill"]

sns.barplot(x=df['size'],y=df['tip_percent'],data=df)

plt.title("Tip Percentage by Dining Party Size")

plt.show()

sns.barplot(x=df['time'], y=df['total_bill'])

plt.title("Highest Total Bill Amount by Time")

plt.show()

df.corr()

sns.heatmap(df.corr(),annot=True)

# OUTPUT
<img width="264" alt="image" src="https://github.com/subikshamalaisamy/DS-EXP-10/assets/87276633/cc643cc3-07c3-47ba-9aa6-ae2a5af2680f">
<img width="94" alt="image" src="https://github.com/subikshamalaisamy/DS-EXP-10/assets/87276633/f4cd2cc8-cc65-4934-9849-1f6b34c3cd61">
<img width="274" alt="image" src="https://github.com/subikshamalaisamy/DS-EXP-10/assets/87276633/7b370a9c-a2a8-4c71-8708-cfb1534058c3">
<img width="279" alt="image" src="https://github.com/subikshamalaisamy/DS-EXP-10/assets/87276633/64ae4fdc-01f4-479c-9c4b-b0e3db99110d">
<img width="341" alt="image" src="https://github.com/subikshamalaisamy/DS-EXP-10/assets/87276633/fa7c33ca-ef4c-4b97-8c96-e308efef3b4b">
<img width="436" alt="image" src="https://github.com/subikshamalaisamy/DS-EXP-10/assets/87276633/57f1823d-f7f1-4f68-b3c8-79c7c7f8ea19">
<img width="427" alt="image" src="https://github.com/subikshamalaisamy/DS-EXP-10/assets/87276633/e854a7ce-473a-4b3f-9d27-8da555a3614d">
<img width="431" alt="image" src="https://github.com/subikshamalaisamy/DS-EXP-10/assets/87276633/36217390-c909-4715-b4da-62f4723200ff">
<img width="355" alt="image" src="https://github.com/subikshamalaisamy/DS-EXP-10/assets/87276633/c3a0b3df-1697-4617-ae68-7c1195135c0b">
<img width="323" alt="image" src="https://github.com/subikshamalaisamy/DS-EXP-10/assets/87276633/d6b26f3c-770d-4e7b-8a7e-087f3e1f22f3">
<img width="343" alt="image" src="https://github.com/subikshamalaisamy/DS-EXP-10/assets/87276633/2a6ab7cb-529c-4ac9-8385-5e3bd1a96d62">
<img width="303" alt="image" src="https://github.com/subikshamalaisamy/DS-EXP-10/assets/87276633/53d81ae3-87ed-408a-ad7c-63387d82ae6b">
<img width="356" alt="image" src="https://github.com/subikshamalaisamy/DS-EXP-10/assets/87276633/e3c9241c-8547-49ed-b088-a2aef4ece109">
<img width="364" alt="image" src="https://github.com/subikshamalaisamy/DS-EXP-10/assets/87276633/6a89d878-13f2-4185-9cf8-36b36f05ca3c">
<img width="358" alt="image" src="https://github.com/subikshamalaisamy/DS-EXP-10/assets/87276633/04b9cfa6-e48d-4a17-a54c-a9d12c6b07ec">
<img width="549" alt="image" src="https://github.com/subikshamalaisamy/DS-EXP-10/assets/87276633/cd996a53-f836-405e-b473-6490202feb08">
<img width="374" alt="image" src="https://github.com/subikshamalaisamy/DS-EXP-10/assets/87276633/d914a8eb-a080-43fb-b2ab-8461cdb83520">

# RESULT
Thus Data Science Process on a complex dataset was performed successfully.

