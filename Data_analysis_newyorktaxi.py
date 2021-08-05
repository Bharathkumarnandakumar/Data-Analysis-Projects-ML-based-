#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
sns.set()


# In[14]:


df = pd.read_csv (r"C:\Users\Ssharma\Desktop\DATA_ENGINEERING\project\UPDATED NEW YRK.csv")


# In[15]:


df.to_json (r'C:/Users/Ssharma/Desktop/DATA_ENGINEERING/updnycjso.json')


# In[16]:


df.columns


# In[17]:


df.dtypes


# In[18]:


df.describe


# In[71]:


df.head


# In[20]:


#fare amount
sns.distplot(df['fare_amount'],kde=False)
plt.title('Distribution of fare amount')
plt.show()


# In[21]:


#Passenger Count
sns.distplot(df['passenger_count'],kde=False)
plt.title('Distribution of Passenger Count')
plt.show()


# In[22]:



sns.distplot(df['trip_distance'],kde=False)
plt.title('The distribution of of the Pick Up  Duration distribution')


# In[76]:


#RELATIONSHIOP BETWEEN vendor id AND TRIP DISTANCE
sns.catplot(x="VendorID", y="trip_distance",kind="strip",data=df)


# In[24]:


#RELATIONSHIOP BETWEEN PASSENGER COUNT AND TRIP DISTANCE
sns.relplot(x="passenger_count", y="trip_distance", data=df, kind="scatter")


# In[53]:


#RELATIONSHIOP BETWEEN distance AND congestion surcharge
sns.relplot(x="congestion_surcharge", y="trip_distance", data=df, kind="scatter")


# In[54]:



#LIBRARIES FOR THE LINEAR REGRESSION_ML MODEL FOR NYC TAXI FARE(DEPENDENT) AND DISTANCE(INDEPENDEDENT)
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score


# In[57]:



#dependent and independent variables
x=df['trip_distance']
y=df['fare_amount']
# Splitting data set into train and test 

x_train, x_test, y_train, y_test = train_test_split(x, y)


# In[59]:


plt.title('Relationship between distance and fare')
plt.scatter(x_train, y_train, color='green')
plt.show()


# In[33]:



#FIT THE MODEL TO DATA
lm=linear_model.LinearRegression()
lm.fit(x_train, y_train)
print('Slope:',lm.coef_)
print('Intercept:',lm.intercept_)


# In[34]:


#EVALUATE THE MODEL
model_score=lm.score(x_train, y_train)
print('Model score:{:.4f}'.format(model_score))


# In[35]:



y_predicted=lm.predict(x_test)

r_squared_score=r2_score(y_test,y_predicted)
print('r square:{:.4f}'.format(r_squared_score))


# In[36]:


plt.title('comparison of y values in test and the predicted values')
plt.plot(x_test,y_predicted,color='black')
plt.scatter(x_test,y_predicted,color='red')
plt.show


# In[84]:



import folium


# In[85]:


map_ny = folium.Map(location=[40.728225, -73.98796844])
map_ny


# In[17]:


import folium
map_1 = folium.Map(location=[40.767937,-73.982155 ],tiles='OpenStreetMap',
 zoom_start=12)
for each in df[:1000].iterrows():
    folium.CircleMarker([each[1]['pickup_latitude'],each[1]['pickup_longitude']],
                        radius=3,
                        color='green',
                        popup=str(each[1]['pickup_latitude'])+','+str(each[1]['pickup_longitude']),
                        fill_color='#FD8A6C'
                        ).add_to(map_1)
map_1


# In[18]:


#Drop-off location for train dataset
import folium # goelogical map
map_3 = folium.Map(location=[40.767937,-73.982155 ],tiles='OpenStreetMap',
 zoom_start=12)
for each in df[:1000].iterrows():
    folium.CircleMarker([each[1]['dropoff_latitude'],each[1]['dropoff_longitude']],
                        radius=3,
                        color='black',
                        popup=str(each[1]['dropoff_latitude'])+','+str(each[1]['dropoff_longitude']),
                        fill_color='#FD8A6C'
                        ).add_to(map_3)
map_3


# In[ ]:




