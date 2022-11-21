#!/usr/bin/env python
# coding: utf-8

# In[2]:


# importing important libraries
# 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.metrics import mean_squared_error


# In[3]:


# importing the data from URL

house_data = pd.read_csv('https://raw.githubusercontent.com/zekelabs/data-science-complete-tutorial/master/Data/house_rental_data.csv.txt', index_col='Unnamed: 0')


# In[4]:


#CONVERTING THE DATA INTO DATA FRAME

df=pd.DataFrame(data=house_data)


# In[5]:


#SHOWING THE 5 ROWS OF DATA

df.head()


# In[6]:


#SHAPE OF DATA 
df.shape


# In[7]:


#COLUMNS AVAILABEL IN DATA
df.columns


# In[8]:


df.info()


# In[9]:


#RENAMING THE LIVING ROOM COLUMN
df.rename(columns={'Living.Room':'LivingRoom'},inplace=True)


# In[10]:


# DESCRIBING THE DATA 
df.describe()


# In[11]:


#CHECKING THE NULL VALUES IN THE DATA 
df.isnull().sum()


# In[12]:


#CREATING FEATURE COLUMN
columns=df.drop(['Price'],axis=1)


# In[13]:


#CREATING FEATURES COLUMNS AS FEATURES 
features=columns


# In[14]:


#CREATING THE TARGET COLUMN
target=house_data['Price']


# In[15]:


#TAKING INFO FOR TARGET
target.info()


# In[16]:


# CHECKING THE SHAPE OF FEATURE AND TARGET 
print(target.shape)
print(features.shape)


# In[17]:


#CHECKING FEATURES INFO 
features.info()


# In[18]:


#CREATING DIFFERENT VARABLE FOR VISUALIZATION WITH TARGET VARIABLE 
sqft=house_data['Sqft']
floor=house_data['Floor']
totalfloor=house_data['TotalFloor']
bedroom=house_data['Bedroom']
livingroom=house_data['LivingRoom']
bathroom=house_data['Bathroom']


# In[19]:


#PLOTTING THE GRAPH BETWEEN SQFT AND TARGET VARIABLE 
plt.xlabel('Feature - X')
plt.ylabel('Target - Y')
plt.scatter(sqft,target)


# In[20]:


plt.xlabel('Feature - X')
plt.ylabel('Target - Y')
plt.scatter(floor,target)


# In[21]:


plt.xlabel('Feature - X')
plt.ylabel('Target - Y')
plt.scatter(totalfloor,target)


# In[22]:


plt.xlabel('Feature - X')
plt.ylabel('Target - Y')
plt.scatter(bedroom,target)


# In[23]:


plt.xlabel('Feature - X')
plt.ylabel('Target - Y')
plt.scatter(livingroom,target)


# In[24]:


plt.xlabel('Feature - X')
plt.ylabel('Target - Y')
plt.scatter(bathroom,target)


# In[25]:


#SPLITTING THE TRAIN AND TEST DATA 

x_train,x_test,y_train,y_test=train_test_split(features,target,test_size=.2,random_state=3)


# In[26]:


#CHECKING THE SSHAPE OF TRAIN AND TEST DATA AND ASSURING IF TRAIN TEST SPLIT PERFORMED WELL.
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[27]:


# IMPORTING THE KNEIGHBORS REGRESSOR 
from sklearn.neighbors  import KNeighborsRegressor


# In[29]:


# FINDING THE OPTIMUM VALUE OF K


rmse_val=[]

for k in range(1,21):
    model=KNeighborsRegressor(n_neighbors=k)
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    error=np.sqrt(mean_squared_error(y_test,y_pred))
    
    rmse_val.append(error)
    
    print('rmse value for ',  k , 'is' ,error)
    
# HERE WE CAN SEE THAT RMSE VALUE FOR 5 IS LOWEST 
# So OPTIMUM VALUE OF K IS CHOSEN AS 5


# In[30]:


#PLOTTING THE GRAPH BETWEEN RMSE AND VALUE OF K,
# SO WE CAN FIND THE OPTIMUM VALUE OF K
# HERE FOR VALUE OF K 5 , RMSE VALUE IS LOWEST.


k_range=range(1,21)
plt.plot(k_range,rmse_val)
plt.xlabel('k')
plt.ylabel('rmse')
plt.show()


# In[31]:


# CREATING A  OPTIMUM MODEL WHERE VALUE OF K IS  5


optimum_model=KNeighborsRegressor(n_neighbors=5)
optimum_model.fit(x_train,y_train)
y_pred1=optimum_model.predict(x_test)
error=np.sqrt(mean_squared_error(y_test,y_pred1))

print('RMSE ' , error)


# In[ ]:




