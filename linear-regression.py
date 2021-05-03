import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression as lr
import  matplotlib.pyplot as plt

data = pd.read_csv("linear.csv")
data.drop('kira', axis=1, inplace = True)

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values = np.NaN, strategy = 'mean', fill_value=None, verbose=0)
newdata = data.iloc[:,0:2].values
print(newdata)

imputer = imputer.fit(newdata[:,0:2])
newdata[:,0:2] = imputer.transform(newdata[:,0:2])
print(newdata)


x=data["metrekare"]
y=data["fiyat"]

x=x.values.reshape(99,1)
y=y.values.reshape(99,1)
lineer_regresyon=lr()
lineer_regresyon.fit(x,y)

lineer_regresyon.predict(x)
m=lineer_regresyon.coef_
b=lineer_regresyon.intercept_

a=np.arange(120)

plt.scatter(x,y)
plt.scatter(a,m*a+b)
plt.show()


