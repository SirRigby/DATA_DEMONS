#!/usr/bin/env python
# coding: utf-8

# ***Importing Libraries***

# In[1]:


import pandas as pd
import numpy as np
import scipy.stats as stats

import datetime

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, precision_score, recall_score, roc_auc_score, roc_curve


# ***Notebook Configration***

# In[2]:


import warnings
warnings.simplefilter('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

pd.set_option('display.float_format', lambda x: '%.2f' % x)

sns.set_style('whitegrid')

get_ipython().run_line_magic('matplotlib', 'inline')


# ***Data Reading***

# In[3]:


df = pd.read_csv(r'C:\Users\piyus\OneDrive\Documents\Notebooks\accidents\Accident.csv', parse_dates=['Date', 'Time'])


# In[4]:


df.columns


# In[5]:


df.drop(columns=['Unnamed: 0', 'Location_Easting_OSGR', 'Location_Northing_OSGR', 
                 'Local_Authority_(Highway)', 'LSOA_of_Accident_Location'], inplace=True)


# In[6]:


df.sample(5)


# ## Data Exploration and Cleaning

# In[7]:


print("No. of rows: {}".format(df.shape[0]))
print("No. of cols: {}".format(df.shape[1]))


# In[8]:


df.info()


# In[9]:


df.isna().any()


# In[10]:


df.isnull().sum() / len(df) * 100


# Hence the null values are very less as compared to the number of rows present in the data, so we can drop the rows 

# In[11]:


df.dropna(subset=['Longitude', 'Time', 'Pedestrian_Crossing-Human_Control', 
                  'Pedestrian_Crossing-Physical_Facilities'], inplace=True)


# In[12]:


dup_rows = df[df.duplicated()]
print("No. of duplicate rows: ", dup_rows.shape[0])


# The total number of the duplicated rows are 34155 hence we can drop these rows

# In[13]:


df.drop_duplicates(inplace=True)
print("No. of rows remaining: ", df.shape[0])


# The followning is the shape of the data after droping the duplicated rows

# In[14]:


df.describe(include=np.number)


# In[15]:


df.describe(include=np.object)


# In[16]:


numerical_data = df.select_dtypes(include='number')
num_cols = numerical_data.columns
len(num_cols)


# In[17]:


categorical_data = df.select_dtypes(include='object')
cat_cols = categorical_data.columns
len(cat_cols)


# In[18]:


sns.set(style="whitegrid")
fig = plt.figure(figsize=(20, 50))
fig.subplots_adjust(right=1.5)

for plot in range(1, len(num_cols)+1):
    plt.subplot(6, 4, plot)
    sns.boxplot(y=df[num_cols[plot-1]])

plt.show()
get_ipython().run_line_magic('time', '')


# In[19]:


import scipy.stats as stats
def diagnostic_plot(data, col):
    fig = plt.figure(figsize=(20, 5))
    fig.subplots_adjust(right=1.5)
    
    plt.subplot(1, 3, 1)
    sns.distplot(data[col], kde=True, color='teal')
    plt.title('Histogram')
    
    plt.subplot(1, 3, 2)
    stats.probplot(data[col], dist='norm', fit=True, plot=plt)
    plt.title('Q-Q Plot')
    
    plt.subplot(1, 3, 3)
    sns.boxplot(data[col],color='teal')
    plt.title('Box Plot')
    
    plt.show()
    
dist_lst = ['Police_Force', 'Accident_Severity',
            'Number_of_Vehicles', 'Number_of_Casualties', 
            'Local_Authority_(District)', '1st_Road_Class', '1st_Road_Number',
            'Speed_limit', '2nd_Road_Class', '2nd_Road_Number',
            'Urban_or_Rural_Area']

for col in dist_lst:
    diagnostic_plot(df, col)
get_ipython().run_line_magic('time', '')


# In[20]:


plt.figure(figsize = (15,10))
corr = df.corr(method='spearman')
mask = np.triu(np.ones_like(corr, dtype=bool))
cormat = sns.heatmap(corr, mask=mask, annot=True, cmap='YlGnBu', linewidths=1, fmt=".2f")
cormat.set_title('Correlation Matrix')
plt.show()


# As from the correlation matrix the columns with high corelation can be obtained

# In[21]:


def get_corr(data, threshold):
    corr_col = set()
    cormat = data.corr()
    for i in range(len(cormat.columns)):
        for j in range(i):
            if abs(cormat.iloc[i, j])>threshold:
                col_name = cormat.columns[i]
                corr_col.add(col_name)
    return corr_col

corr_features = get_corr(df, 0.80)
print(corr_features)


# These is only one column with corelation more than 80% hence we can drop that following columns

# In[22]:


df.drop(columns=['Local_Authority_(District)'], 
        axis=1, inplace=True)


# In[23]:


def pie_chart(data, col):

  x = data[col].value_counts().values
  plt.figure(figsize=(7, 6))
  plt.pie(x, center=(0, 0), radius=1.5, labels=data[col].unique(), 
          autopct='%1.1f%%', pctdistance=0.5)
  plt.axis('equal')
  plt.show()

pie_lst = ['Did_Police_Officer_Attend_Scene_of_Accident']
for col in pie_lst:
  pie_chart(df, col)


# In[24]:


def cnt_plot(data, col):

  plt.figure(figsize=(15, 7))
  ax1 = sns.countplot(x=col, data=data,palette='rainbow')

  for p in ax1.patches:
    ax1.annotate('{}'.format(p.get_height()), (p.get_x()+0.15, p.get_height()+1), ha='center')

  plt.show()

  print('\n')

cnt_lst1 = ['Road_Type', 'Junction_Control',
           'Pedestrian_Crossing-Human_Control',
           'Road_Surface_Conditions']

for col in cnt_lst1:
  cnt_plot(df, col)
get_ipython().run_line_magic('time', '')


# In most of the accidents,
# 
# The road was single carriageway.
# 
# The junction was either uncontrolled junction or there wasn't any junction at all.
# 
# There were no human controlled pedestrian crossing within 50 metres of the spot.
# 
# The weather was dry.

# In[25]:


def cnt_plot(data, col):

  plt.figure(figsize=(10, 7))
  sns.countplot(y=col, data=data,palette='rainbow')
  plt.show()

  print('\n')
  
cnt_lst2 = ['Pedestrian_Crossing-Physical_Facilities', 'Light_Conditions',
            'Weather_Conditions',
            'Special_Conditions_at_Site', 'Carriageway_Hazards']

for col in cnt_lst2:
  cnt_plot(df, col)
get_ipython().run_line_magic('time', '')


# In most of the accidents,
# 
# There was no physical crossing within 50 metres of the spot.
# 
# Happened in daylight so the visibility was fine.
# 
# The weather was fine without high winds.
# 
# There wasn't any special condition or any problem with the carriageway.

# In[26]:


df.sample(5)


# In[27]:


df['Urban_or_Rural_Area'].value_counts()


# In[28]:


df['Urban_or_Rural_Area'].replace(3, 1, inplace=True)


# In[29]:


df['Accident_Severity'].value_counts()


# In[30]:


df['Number_of_Vehicles'].value_counts()[:10]


# In[31]:


df['Number_of_Casualties'].value_counts()[:10]


# In[32]:


dt1 = df.groupby('Date')['Accident_Index'].count().reset_index().rename(columns={'Accident_Index':'No. of Accidents'})

fig = px.line(dt1, x='Date', y='No. of Accidents',
              labels={'index': 'Date', 'value': 'No. of Accidents'})
fig.show()


# In[33]:


dt2 = df.groupby('Year')['Accident_Index'].count().reset_index().rename(columns={'Accident_Index':'No. of Accidents'})

fig = px.line(dt2, x='Year', y='No. of Accidents',
              labels={'index': 'Year', 'value': 'No. of Accidents'})
fig.show()


# In[34]:


dt3 = df.groupby('Day_of_Week')['Accident_Index'].count().reset_index().rename(columns={'Accident_Index':'No. of Accidents'})

fig = px.line(dt3, x='Day_of_Week', y='No. of Accidents',
              labels={'index': 'Day_of_Week', 'value': 'No. of Accidents'})
fig.show()


# In[35]:


cat_cols


# In[36]:


len(df['Accident_Index'].unique())
df.drop('Accident_Index',axis=1,inplace=True)


# Beacause the feature is just a count we can drop the feature (The following feature adds no value to the model)

# In[37]:


df.head()


# In[38]:


plt.figure(figsize=(18, 5))
sns.heatmap(df.isna(), yticklabels=False, cbar=False, cmap='viridis')
plt.show()
get_ipython().run_line_magic('time', '')


# In[39]:


X = df.drop(columns=['Road_Type'], axis=1)

plt.figure(figsize=(8, 10))
X.corrwith(df['Accident_Severity']).plot(kind='barh', 
                               title="Correlation with 'Convert' column -")
plt.show()


# The Feature Accident Severity is highly correlated with the feature like "Number of Vechiles", "Numbber of casulaity" and "Speed Limit"

# ## Feature Engineering and Scaling

# In[40]:


cat_cols=[feature for feature in df.columns if df[feature].dtype=='O']
print(cat_cols)


# In[41]:


for feature in cat_cols:
    print(f'The {feature} has following number of {len(df[feature].unique())}')


# Hence the catgorical variables have values ranging from 2 to 9, we can use LabelEncoder

# ***Label Encoding***

# In[42]:


from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()


# In[43]:


for feature in cat_cols:
    df[feature]=labelencoder.fit_transform(df[feature])


# In[44]:


df.head()


# In[45]:


df.drop('Year',axis=1,inplace=True)


# In[46]:


df["day"] = df['Date'].map(lambda x: x.day)
df["month"] = df['Date'].map(lambda x: x.month)
df["year"] = df['Date'].map(lambda x: x.year)


# Converting the DateTime Variables to be feeded to the Model

# In[47]:


df.head()


# In[48]:


df.drop("Date",axis=1,inplace=True)
df.drop("Time",axis=1,inplace=True)


# In[49]:


df['Accident_Severity']=df['Accident_Severity'].map({1:0,2:1,3:2})


# ## Scaling the Data

# ***Checking feature importance***

# In[50]:


from sklearn.preprocessing import StandardScaler
features = [feature for feature in df.columns if feature!='Accident_Severity']
xx = df.loc[:, features]
yy = df.loc[:,['Accident_Severity']]


# In[51]:


from sklearn.ensemble import ExtraTreesClassifier
clf = ExtraTreesClassifier()
clf.fit(xx, yy)


# In[52]:


get_ipython().run_line_magic('time', '')
imp = clf.feature_importances_


# In[53]:


feature_importances = pd.Series(imp, index=xx.columns)


# In[54]:


top10_feat = feature_importances.nlargest(10)


# In[55]:


top10_feat


# In[56]:


dfnew=df[['Latitude','Longitude','day','month','1st_Road_Number','year','Day_of_Week','Accident_Severity']]


# In[57]:


dfnew.head()


# In[58]:


from sklearn.preprocessing import StandardScaler
features = [feature for feature in dfnew.columns if feature!='Accident_Severity']
x = dfnew.iloc[0:50000, :-1]
y = dfnew.iloc[0:50000,[-1]]
x = StandardScaler().fit_transform(x)


# ## Using SMOTE to Treat imbalenced Dataset

# In[59]:


get_ipython().run_line_magic('time', '')
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split


from imblearn.over_sampling import SMOTE
oversample = SMOTE()
x, y = oversample.fit_resample(x, y)


# In[60]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# ## Using XGB Classifier for Prediction

# In[61]:


from xgboost import XGBClassifier


# In[62]:


xgb = XGBClassifier(n_estimators=100)


# In[63]:


get_ipython().run_line_magic('time', '')
xgb.fit(x_train, y_train)


# In[64]:


get_ipython().run_line_magic('time', '')
preds = xgb.predict(x_test)


# In[65]:


from sklearn.metrics import accuracy_score,classification_report
score=accuracy_score(preds,y_test)
print(score)


# In[66]:


print(classification_report(preds,y_test))


# In[69]:


import pickle
pickle.dump(xgb, open('modelxgblast.pkl', 'wb'))


# In[70]:


pickled_model = pickle.load(open('modelxgblast.pkl', 'rb'))
new=pickled_model.predict(x_test)


# In[71]:


conmat = confusion_matrix(y_test, preds)
sns.heatmap(conmat, annot=True, cbar=False)
plt.title("Confusion Matrix")
plt.show()

