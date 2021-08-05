#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib 
matplotlib.rcParams["figure.figsize"] = (20,10)


# In[2]:


import xlrd


# In[3]:


from pandas_profiling import ProfileReport


# In[4]:


#extraction
df2 = pd.read_csv("C:/Users/bhara/Desktop/data curactio/1Bangalore_House_Data.csv")
df2.head()


# In[5]:


#data profiling
ProfileReport(df2)


# In[6]:


df2.columns


# In[7]:


#Handling missing values to achieve COMPLETENESS
df2.isnull().sum()


# In[8]:


df3 = df2.dropna()
df3.isnull().sum()


# In[9]:



df3.shape


# In[10]:


#removing special characters to achieve accuracy
df3.columns = df3.columns.str.replace('[#,@,%,^]', '')


# In[11]:


df3


# In[12]:


#HANDLING Inconsistent values to achieve consistency

df3.columns = df3.columns.str.title()


# In[13]:


df3


# In[14]:


df3['bhk'] = df3['Size'].apply(lambda x: int(x.split(' ')[0]))
df3.bhk.unique()


# In[15]:


def is_float(x):
    try:
        float(x)
    except:
        return False
    return True


# In[16]:


#consistency
df3[~df3['Total_Sqft'].apply(is_float)]


# In[ ]:


#DIMENSIONALITY REDUCTION,REMOVING OUTLIERS ,FEATURE ENGG,ONE HOT ENCODING AND PREPROCESSING PROCESS REQUIRED FOR ML QUESTION


# In[17]:



def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None


# In[18]:


df4 = df3.copy()
df4.Total_Sqft = df4.Total_Sqft.apply(convert_sqft_to_num)
df4 = df4[df4.Total_Sqft.notnull()]
df4.head(2)


# In[19]:


df4.loc[30]


# In[20]:


(2100+2850)/2


# In[21]:


df5 = df4.copy()
df5['price_per_sqft'] = df5['Price']*100000/df5['Total_Sqft']
df5.head()


# In[22]:


df5_stats = df5['price_per_sqft'].describe()
df5_stats


# In[23]:


df5.to_csv("C:/Users/bhara/Desktop/hello/output_cleaned/final.csv",index=False)


# In[ ]:





# In[24]:


df5.Location = df5.Location.apply(lambda x: x.strip())
location_stats = df5['Location'].value_counts(ascending=False)
location_stats


# In[25]:


location_stats.values.sum()


# In[26]:


#DIMENSIONALITY REDUCTION
location_stats_less_than_10 = location_stats[location_stats<=10]
location_stats_less_than_10


# In[27]:


df5.Location = df5.Location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
len(df5.Location.unique())


# In[28]:


df5[df5.Total_Sqft/df5.bhk<300].head()


# In[29]:


df6 = df5[~(df5.Total_Sqft/df5.bhk<300)]


# In[30]:



def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('Location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
df7 = remove_pps_outliers(df6)
df7.shape


# In[31]:


def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('Location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df8 = remove_bhk_outliers(df7)
# df8 = df7.copy()
df8.shape


# In[32]:


df8.Bath.unique()


# In[33]:


df9 = df8[df8.Bath<df8.bhk+2]
df9.shape


# In[34]:


df9.head(2)


# In[35]:


df10 = df9.drop(['Size','price_per_sqft'],axis='columns')
df10.head(3)


# In[36]:


dummies = pd.get_dummies(df10.Location)
dummies.head(3)


# In[37]:


df11 = pd.concat([df10,dummies.drop('other',axis='columns')],axis='columns')
df11.head()


# In[38]:


df12 = df11.drop('Location',axis='columns')
df12.head(2)


# In[39]:


df12.shape


# In[40]:


X = df12.drop(['Price'],axis='columns')
X.head(3)


# In[41]:


X.shape


# In[42]:


y = df12.Price
y.head(3)


# In[ ]:





# In[55]:


#star schema
# Import Libraries required for Processing
import pandas as pd
import numpy as np

# Setting file names used during execution process
# Input File :  Movie details and User Information Excel file consists of raw information about User
#               and Movies along with the ratings provided by User to a specific Movie
file_input = "C:/Users/bhara/Desktop/Bangalore_House_Data1.xls"

# Output File : Pre-Processed Movie Lens Data CSV file will hold the information, after input file
#               is being pre-processed; so as to remove Null Value and InConsistent value.
file_pre_processed = "C:/Users/bhara/Desktop/hello/output_cleaned/final.csv"

# Output File : User_Details.xlsx file will hold only the User Information.
#               Movie_Details.xlsx file will hold only the Movie Information.
#               Movie_Ratings_for_Users.xlsx file will hold the factual information of 
#               Movie Ratings provided by User.
file_location_output = "C:/Users/bhara/Desktop/data curactio/4location5.csv"
file_house_output = "C:/Users/bhara/Desktop/data curactio/4house6.csv"
file_cost_output = "C:/Users/bhara/Desktop/data curactio/4cost8.csv"
fact_table = "C:/Users/bhara/Desktop/data curactio/4fact9.csv"

# Read Input File to fetch Raw Information of User and Ratings provided to specific movie 

# Read the Movie details and User Information Excel file using pandas.read_excel function and 
# populate the records into an input Dataframe
# Link: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_excel.html
df_input = pd.read_excel(file_input, header = 0)


# Creation of Star Schema

# Read Pre-processed Movie Lens Data CSV File which is populated above using pandas.read_csv function 
# and populate the records into a new dataframe (df_pre_processed) 
df_pre_processed = pd.read_csv(file_pre_processed, header = 0)

# Split the columns as per requirement from df_pre_processed dataframe into different dataframes.
df_pre_processed 


# In[98]:


# Select columns which are relevant to user information from df_pre_processed dataFrame
# and populate it into a new subset DataFrame (df_users)
df_location = df_pre_processed.loc[ : , ['Location','Total_Sqft'] ]
df_location


# In[99]:


# Select columns which are relevant to movie information from df_pre_processed dataFrame
# and populate it into a new subset DataFrame (df_movies)
df_house_dimension = df_pre_processed.loc[ : , ['Bath','bhk','Size'] ]
df_house_dimension


# In[100]:


# Select columns which are relevant to factual information about Movie Rating from df_pre_processed
# dataFrame and populate it into a new subset DataFrame (df_user_movie_fact)
df_cost = df_pre_processed.loc[ : , ['Price', 'price_per_sqft'] ]

df_cost


# In[101]:


df_fact = df_pre_processed.loc[ : , ['Location'] ]

df_fact


# In[102]:


# df_users dataframe represent User Dimension Table,
# df_movies dataframe represent Movie Dimension Table,
# df_user_movie_fcat represent Movie Rating Fact Table.

# Requirement states that all table must have unique records and they must be sorted.
# Export the Star Schema and save them in Excel File respectively
df_location.to_csv(file_location_output,index=False)
df_house_dimension.to_csv(file_house_output ,index=False)
df_cost.to_csv(file_cost_output,index=False)

df_fact.to_csv(fact_table,index=False)


# In[ ]:


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


#data analytics part


# In[44]:


import openpyxl


# In[45]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)


# In[46]:


from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(X_train,y_train)
lr_clf.score(X_test,y_test)


# In[47]:



from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

cross_val_score(LinearRegression(), X, y, cv=cv)


# In[48]:



from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

def find_best_model_using_gridsearchcv(X,y):
    algos = {
        'linear_regression' : {
            'model': LinearRegression(),
            'params': {
                'normalize': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1,2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion' : ['mse','friedman_mse'],
                'splitter': ['best','random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores,columns=['model','best_score','best_params'])

find_best_model_using_gridsearchcv(X,y)


# In[49]:


def predict_price(location,sqft,bath,bhk):    
    loc_index = np.where(X.columns==location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return lr_clf.predict([x])[0]


# In[50]:


predict_price('1st Phase JP Nagar',1000, 2, 2)


# In[51]:


predict_price('1st Phase JP Nagar',1000, 3, 3)


# In[52]:


predict_price('Indira Nagar',1000, 3, 3)


# In[53]:


predict_price('Indira Nagar',1000, 2, 2)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




