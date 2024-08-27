program 1

import pandas as pd
data=pd.read_csv("/content/iris.csv")
print(data.head())

print(data.sample(10))

print(data.columns)

print(data.shape)

sliced_data=data[10:21]

print(sliced_data)

print(data.iloc[5])

print(data.loc[data["species"]=="setosa"])

program 2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
data = pd.DataFrame(data=iris['data'], columns=iris['feature_names'])
data['species'] = iris['target']

species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
data['species'] = data['species'].map(species_map)

plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='sepal length (cm)', y='sepal width (cm)', hue='species',palette='viridis')
plt.title('Sepal Length vs. Sepal Width')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.legend(title='Species')
plt.show()

plt.figure(figsize=(12,8))
sns.boxplot(data=data, x='species', y='petal length (cm)', palette='viridis')
plt.title('Box Plot of Petal Length by Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.show()

sns.pairplot(data, hue='species', palette='viridis')
plt.suptitle('Pair Plot of Iris Dataset', y=1.02)
plt.show()

plt.figure (figsize=(12,8))
sns.violinplot(data=data, x='species', y='petal width (cm)', palette='viridis')
plt.title('Violin Plot of Petal Width by Species')
plt.xlabel('Species')
plt.ylabel('Petal Width (cm)')
plt.show()

correlation_matrix=data.iloc[:,:-1].corr()

plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot=True, cmap='viridis', linewidths=0.5)
plt.title('heatmap of feature correlations in iris dataset')
plt.show()

progarm 3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

data = pd.read_csv('/content/train.csv', encoding='latin1')
df = pd.DataFrame(data)
data.info()
data.describe()
data.head()
sns.pairplot(data)

np.random.seed(42)
num_customers=100
data={
    'Customer_id':np.arange(1,num_customers+1),
    'Gender':np.random.choice(['Male','Female'],num_customers),
    'Age':np.random.randint(18,70,num_customers),
    'Income':np.random.randint(20000,150000,num_customers),
    'Speding_Score':np.random.randint(1,101,num_customers)
}

df=pd.DataFrame(data)

df.to_csv('data.csv',index = False)
data=pd.read_csv('datapre.csv')

imputer=SimpleImputer(strategy='mean')
data['age']=imputer.fit_transform(data[['age']])
data

le=LabelEncoder()
data['Gender']=le.fit_transform(data['Gender'])
data

scaler=StandardScaler()
data[['Age','Income']]=scaler.fit_transform(data[['Age','Income']])
data

outliers =data[np.abs(data['z-score'])>3]
print(outliers)

Q1=data['Age'].quantile(0.25)
Q3=data['Age'].quantile(0.75)
IQR=Q3-Q1
outliers=data[(data['Age']<(Q1-1.5*IQR))|(data['Age']>(Q3+1.5*IQR))]
print("outliers:")
print(outliers)

plt.scatter(data['Age'],data['Income'])
plt.title('Scatter Plot')
plt.xlabel('Age')
plt.ylabel('Income')
plt.show()

data_cleaned=data[(np.abs(data['z-score'])<=3)]
data_cleaned=data_cleaned.drop(columns=['z-score'])
data

from sklearn.model_selection import train_test_split
X=data[['Age','Gender']]
Y=data['Income']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
print("X_train:")
print(X_train)
print("X_test:")
print(X_test)
print("Y_train:")
print(Y_train)
print("Y_test:")
print(Y_test)

program 4
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = pd.read_csv('/content/train.csv', encoding='latin1')
df = pd.DataFrame(data)
df.head(10)

df.fillna(df.mean(), inplace=True)
df.isnull().sum()

df.drop_duplicates(inplace=True)
df.duplicated().sum()

from scipy import stats
z_scores = np.abs(stats.zscore(df[['x']]))
df = df[(z_scores < 3).all(axis=1)]

x = np.array(x).reshape(-1, 1)
y = np.array(y).reshape(-1, 1)

model = LinearRegression()
model.fit(x, y)
predicted_scores = model.predict(x)

print("Coefficient: ", model.coef_)
print("Intercept: ", model.intercept_)
plt.scatter(x, y)
plt.plot(x, predicted_scores, color='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression Model')
plt.show()

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

y_pred = model.predict(x)
y_true = y

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print("Mean Squared Error (MSE): ", mse)
print("Root Mean Squared Error (RMSE): ", rmse)
print("Mean Absolute Error (MAE): ", mae)
print("R-squared (R2): ", r2)
