#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("concrete_data.csv")


# In[3]:


df.head()


# In[4]:


df.columns


# In[5]:


name_col = df.columns.tolist()


# In[6]:


name_col


# In[7]:


name_col[0]


# In[8]:


df.info()


# In[9]:


df.describe()


# In[10]:


df.isna().sum()


# In[11]:


df.duplicated().sum()


# In[12]:


df[df.duplicated() == True]


# In[13]:


df.drop_duplicates(keep = 'first', inplace = True)
df.duplicated().sum()


# In[14]:


df


# In[15]:


df.reset_index(drop = True, inplace = True)
df


# In[16]:


plt.figure(figsize = (15,15), facecolor = 'white')
plotnumber = 1
for i in df.columns:
    ax = plt.subplot(4,3,plotnumber)
    sns.histplot(df[i])
    plt.xlabel(i, fontsize = 10)
    plotnumber +=1
plt.tight_layout()# use of tight layout in order to remove white spaces and make all plot compact and together
plt.show()


# In[17]:


plt.figure(figsize = (15,15), facecolor = 'white')
plotnumber = 1
for i in df.columns:
    ax = plt.subplot(4,3,plotnumber)
    sns.boxplot(df[i])
    plt.xlabel(i, fontsize = 10)
    plotnumber +=1
plt.tight_layout()# use of tight layout in order to remove white spaces and make all plot compact and together
plt.show()


# In[18]:


df.columns


# In[19]:


outliers = ['blast_furnace_slag','water','superplasticizer','concrete_compressive_strength' ,'age']


# In[20]:


def outlier_capping(dataframe: pd.DataFrame, outliers:list):
    df = dataframe.copy()
    for i in outliers:
        q1 = df[i].quantile(0.25)
        q3 = df[i].quantile(0.75)
        iqr = q3 - q1
        upper_limit = q3 + 1.5*iqr
        lower_limit = q3 - 1.5*iqr
        df.loc[df[i]>upper_limit,i] = upper_limit
        df.loc[df[i]<lower_limit,i] = lower_limit
    return df   
df = outlier_capping(dataframe = df, outliers = outliers)


# In[21]:


plt.figure(figsize = (15,15), facecolor = 'white')
plotnumber = 1
for i in df.columns:
    ax = plt.subplot(4,3,plotnumber)
    sns.boxplot(df[i])
    plt.xlabel(i, fontsize = 10)
    plotnumber +=1
plt.tight_layout()# use of tight layout in order to remove white spaces and make all plot compact and together
plt.show()


# In[22]:


X = df.drop('concrete_compressive_strength', axis = 1)
y = df['concrete_compressive_strength']


# In[23]:


plt.figure(figsize = (15,15), facecolor = 'white')
plotnumber = 1
for i in X.columns:
    ax = plt.subplot(4,3,plotnumber)
    sns.scatterplot(x = df[i], y = y)
    plt.xlabel(i, fontsize = 10)
    plotnumber +=1
plt.tight_layout()# use of tight layout in order to remove white spaces and make all plot compact and together
plt.show()


# In[24]:


plt.figure(figsize = (10,10))
sns.heatmap(df.corr(), annot = True)
plt.show()


# In[39]:


from sklearn.model_selection import train_test_split

# xtrain, xtest , ytest, ytrain = train_test_split(X , y , test_size = 0.3, random_state = 42)
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=42)


# In[40]:


from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score


# In[42]:


def check_model_performance(preprocessor, xtrain , ytrain, xtest, ytest):
    models = {'Linear Regression': LinearRegression(),
             'Ridge Regression':Ridge(alpha = 1),
             'Lassor Regression ': Lasso(alpha = 1),
             'Random Forest Regression': RandomForestRegressor(max_depth = 5),
             'Gradient Boosting Regression': GradientBoostingRegressor(learning_rate= 0.1)}
    for model_name, model in models.items():
        pipeline = make_pipeline(preprocessor, model)
        pipeline.fit(xtrain, ytrain)
        y_pred = pipeline.predict(xtest)
        mse = mean_squared_error(ytest, y_pred)
        r2 = r2_score(ytest, y_pred)
        print(f"{model_name} - Mean Squared Error = {mse} \n{model_name} - r2_scored = {r2}")


# In[43]:


from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline

def check_model_performance(preprocessor, xtrain, ytrain, xtest, ytest):
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1),
        'Lasso Regression': Lasso(alpha=1),  # Corrected the model name
        'Random Forest Regression': RandomForestRegressor(max_depth=5),
        'Gradient Boosting Regression': GradientBoostingRegressor(learning_rate=0.1)
    }

    for model_name, model in models.items():
        pipeline = make_pipeline(preprocessor, model)
        pipeline.fit(xtrain, ytrain)
        y_pred = pipeline.predict(xtest)
        mse = mean_squared_error(ytest, y_pred)
        r2 = r2_score(ytest, y_pred)
        print(f"{model_name} - Mean Squared Error = {mse} \n{model_name} - R2 Score = {r2}")

# Example usage:
# Assuming you have preprocessor, xtrain, ytrain, xtest, and ytest defined
# check_model_performance(preprocessor, xtrain, ytrain, xtest, ytest)


# In[44]:


preprocessor_01 = make_pipeline(KNNImputer(n_neighbors=3),StandardScaler())
preprocessor_02 = make_pipeline(KNNImputer(n_neighbors=3),MinMaxScaler())
preprocessor_03 = make_pipeline(KNNImputer(n_neighbors=3),RobustScaler())

print(f"{'=' * 10} Result for StandardScaler {'=' *10}")
check_model_performance(preprocessor_01, xtrain, ytrain, xtest, ytest)

print(f"\n{'=' * 10} Result for MinMaxScaler {'=' *10}")
check_model_performance(preprocessor_02, xtrain, ytrain, xtest, ytest)

print(f"\n{'=' * 10} Result for RobustScaler {'=' *10}")
check_model_performance(preprocessor_03, xtrain, ytrain, xtest, ytest)


# In[45]:


from sklearn.model_selection import GridSearchCV
param_grid = {'n_estimators' : [100,200],
             'learning_rate' : [0.1, 0.01],
             'max_depth': [5,3,7],
             'min_samples_split' : [2,4],
             'min_samples_leaf' : [1,2,3]}

gb_rg = GradientBoostingRegressor()

grid = GridSearchCV(gb_rg, param_grid, scoring = 'neg_mean_squared_error', cv = 5, verbose = 1)
grid.fit(xtrain, ytrain)


# In[46]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor

# Your dataset (assuming xtrain and ytrain are defined)
# xtrain, ytrain = ...

param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.1, 0.01],
    'max_depth': [5, 3, 7],
    'min_samples_split': [2, 4],
    'min_samples_leaf': [1, 2, 3]
}

gb_rg = GradientBoostingRegressor()

grid = GridSearchCV(
    gb_rg,
    param_grid,
    scoring='neg_mean_squared_error',
    cv=5,
    verbose=1
)

grid.fit(xtrain, ytrain)



# In[47]:


grid.best_params_


# In[48]:


grid.best_score_


# In[51]:


grid.best_estimator_


# In[50]:


grid.best_estimator_.score(xtest, ytest)


# In[53]:


ypred = grid.best_estimator_.predict(xtest)
mean_squared_error(ytest, ypred)


# In[ ]:




