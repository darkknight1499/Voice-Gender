import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

#read data file
data = pd.read_csv("voice.csv")
#print(data.head())  # will give top five rows

#data.tail() # give last five rows

data.shape # print shape(row, column)

data.info()  # check what kind of data are

# we have object type only label which we need to encode for machine
# learning models

# check data has any values null/nan or not# check
v = data.isnull().values.any()
print(v)

#It's good sign we dont have to do fill empty places
column_names = data.columns.tolist()
print(column_names)   # print columns name

# checking Multicolinearity betwwen each and every attribute
correlation = data.corr()
print(correlation)

# plot correlation matrix
f, ax = plt.subplots(figsize = (8,8))
# Draw the heatmap using seaborn
seaborn.heatmap(correlation, square = True)
plt.show()

#box plot represetation
data.boxplot(column='sd',by='label', grid=False)
plt.show()

data.boxplot(column= 'median', by='label', grid=False)
plt.show()

male=data[data['label'] == 'male'].shape[0]
print('Male:',male)

female=data[data['label'] == 'female'].shape[0]
print('Female:',male)

# for checking difference between mal and female
a = data[data['label'] == 'male'].mean()
print(a)
b = data[data['label'] == 'female'].mean()
print(b)


#Distribution of male and female it's another way of box plot #Distrib
seaborn.FacetGrid(data, hue="label", height=6) \
   .map(seaborn.kdeplot, "meanfun") \
   .add_legend()
plt.show()

#Distribution of male and female
seaborn.FacetGrid(data, hue="label", height=6) \
   .map(seaborn.kdeplot, "meanfreq") \
   .add_legend()
plt.show()

seaborn.FacetGrid(data, hue="label", height=6) \
   .map(seaborn.kdeplot, "IQR") \
   .add_legend()
plt.show()

data.plot(kind='scatter', x='meanfreq', y='dfrange')
plt.show()
data.plot(kind='kde', y='meanfreq')
plt.show()

# convert srting data into numberic eg. male 1, female 0
df = data
df = df.drop(['label'],axis = 1)
X = df.values
y = data['label'].values

# only one column has object type so we encode it

encoder = LabelEncoder()
y = encoder.fit_transform(y)
#print(y)


# convert srting data into numberic eg. male 1, female 0
df1 = data
df1 = df1.drop(['label'],axis = 1)
X1 = df1.values
y1 = data['label'].values

# only one column has object type so we encode it

encoder = LabelEncoder()
y1 = encoder.fit_transform(y)

X_train,X_test,y_train,y_test =  train_test_split(X1, y1, test_size=0.2, random_state=1)


#KNN Implementation
clf = KNeighborsClassifier()
clf.fit(X_train, y_train)
y_pred2 = clf.predict(X_test)
print('KNN without Preprocessed Data:',metrics.accuracy_score(y_test, y_pred2))

print('============After performing The StandardScaler As preprocessing of data============')
scaler = StandardScaler()
scaler.fit(X1)
X1 = scaler.transform(X1)
print(X1)

X_train,X_test,y_train,y_test =  train_test_split(X1, y1, test_size=0.2, random_state=1)
#KNN Implementation
clf = KNeighborsClassifier()
clf.fit(X_train, y_train)
y_pred2 = clf.predict(X_test)
print('KNN with Preprocessed data:',metrics.accuracy_score(y_test, y_pred2))
