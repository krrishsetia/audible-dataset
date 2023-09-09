import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sklearn as sl
from sklearn import metrics
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


pd.options.display.max_columns = 2
pd.options.display.max_rows = 100000
pd.options.display.max_columns = 2
pd.options.display.max_rows = 100000
data = pd.read_csv('csv files/audible_uncleaned.csv')

def float_conversion(var):
    a = var.replace(',','')
    if var == 'Free':
        return(0)
    else:return(float(a))

data['price'] = data['price'].apply(float_conversion)

def str_time_check(var):
    a = ''.join((char for char in var
                           if char.isalpha() or char.isspace()))
    if a == ' hrs and  mins' or a == ' hr and  mins' or a == ' hrs and  min' or a == ' hr and  min':
        b = ''.join((char for char in var
                           if char.isnumeric() or char.isspace()))
        c = (int(b[0:2]) * 60) + int(b[4:7])
        return c

    elif a == ' mins' or a == ' min' :
        b = ''.join((char for char in var
                     if char.isnumeric() or char.isspace()))
        c = int(b[0:2])
        return c

    elif a == ' hrs' or a == ' hr':
        b = ''.join((char for char in var
                     if char.isnumeric() or char.isspace()))
        c = int(b[0:2]) * 60
        return c
    elif a == ('Less than  minute'):
        c = 1
        return c

data['time'] = data['time'].apply(str_time_check)
temp = []
def str_stars_check(var):
    a = ''.join((char for char in var
                           if char.isalpha() or char.isspace()))
    if a == ' out of  stars rating' or a == ' out of  stars ratings':
        b = ''.join((char for char in var
                           if char.isnumeric() or char.isspace()))
        if len(b) >= 9:
            e =b[0]+'.'+b[1:3]
        else: e = b[0:3]
        c = float(e)
        d = int(b[6:10])
        temp.append(d)
        return c
    elif a == ('Not rated yet'):
        c = 0
        d = 0
        temp.append(d)
        return c
data['stars'] = data['stars'].apply(str_stars_check)
df = pd.Series(data=temp,name='reviews')
data.drop(['name','author','narrator','language'],inplace=True,axis=1)
data['reviews'] = df

x = data['stars'].values.reshape(-1,1)
y = data['time'].values.reshape(-1,1)

x_train,x_test,y_train,y_test = train_test_split(x,y)
lr = LinearRegression()
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)

sns.displot(data,x='stars',y='time')
plt.plot(x_test,y_pred)
plt.show()