#!/usr/bin/env python
# coding: utf-8

# # Credit Amount EDA Case Study

# ![loan-2.png](attachment:loan-2.png)

# ### Introduction

# We have loan applications data for about 307k applications. The goal of this case is to perform Risk Analytics with the help of data wrangling and visualisation libraries of Python. The end goal is to derive important insights for the bank to identify the characteristics for bad loan applications. ( Bad loans are loans which are delayed/not paid.)

# ### Objectives

# * Identify what are some common characteristics of bad loan applications
# * Identify if there are any patterns related to applicants with loan difficulties
# * Identify the driving factors or strong indicators of a bad loan application

# ### Data Dictionary 

# A common starting point in any EDA problem is - Understanding the data. 
# 
# The first step is to check if there is a data dictionary availalble, and try to get a good understanding of the 
# level of the data and meaning of each of the columns.
# 
# The data dictionary document has been provided along with the data. It is advised to go through each of the column in 
# data once before starting with EDA.
# 
# A screenshot of the same is given below - 

# ![image.png](attachment:image.png)

# The dictionary can also be imported to Jupyter notebook as below - 

# ##### Changing the current working directory to fetch the .csv file-

# In[1]:


import os
#os.chdir("C:\\Users")
print('Current working directory is ----    '+ os.getcwd())
# os.chdir("D:\\Relevel\\Week 13_Day4")
print('Changed working directory is ----    '+ os.getcwd())


# In[2]:


import pandas as pd


# In[3]:


# NOTE : New function

# Extending the default setting for maximum displayable rows/columns in Jupyter
# If not used, Jupyter wll truncate some rows and columns by default. 

pd.options.display.max_rows = 4000
pd.options.display.max_columns = 1000


# In[4]:


data_dict = pd.read_csv("columns_description - columns_description.csv")
data_dict.head(100)


# # EDA - Credit Applications 

# Let's begin our EDA now. The flow of the entire case would be as follows - 
# 
# 1. Data Wrangling
# 2. Univariate Analysis
# 3. Bivariate/Multivariate Analysis
# 4. Final Insights

# ## Importing libraries

# In[5]:


# For suppressing warning messages
import warnings

warnings.filterwarnings('ignore')


# In[3]:


#importing core libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


## Changing default figure size using rcParams

print("Earlier, figure default size was: ", plt.rcParams["figure.figsize"])
plt.rcParams["figure.figsize"] = (10, 5)
print("Now, figure default size is: ", plt.rcParams["figure.figsize"])


# In[8]:


# Setting theme for charts

plt.style.use('ggplot')


# ## 1. Data Wrangling 

# Loading the data - 

# In[9]:


credit_data = pd.read_csv("application_data.csv")
credit_data.head(20)


# We see that the data is at Loan ID level ( SK_ID_CURR).
# There is a mix of quantitative and qualitative variables.
# There are a lot of Flags as well. 
# There are considerable NAs as well at the first glance of data.

# In[10]:


# Create a plot of how many Men and Women are there

# which column - CODE_GENDER
# which plot type - bar plot
# <>

sns.countplot('CODE_GENDER', data=credit_data)


# ### 1.1 Inspecting data

# In[11]:


# Checking shape of the data

credit_data.shape


# Now we know that there are 307511 loan applications and 122 fields for each application. 

# In[12]:


credit_data.info()     # Overview and schema of dataset


# In[ ]:


credit_data.describe()    # descriptive statistics


# In[13]:


credit_data.shape


# ##### 5 Point summary

# describe function is used to get the 5 number statistical summary of the quantitative variables of a data.
# The focus points could be the Range, mean and median for each variable to get a better understanding of the variables.
# 
# ( Do a quick search about 5 Point summary in case not aware )

# In[17]:


# Checking 5 point summary with describe function

credit_data.describe()


# ##### Invalid -ve values

# Another important use case of describe function is to check for Invalid -ve values.
# Here for example, a close look will tell us that all the columns starting with "DAYS_..." (example - DAYS_BIRTH ) have -ve values which cannot be valid.
# We will clean the data by tranforming this data appropriately later. 
# 

# ##### Getting the list of % Nulls in each column

# In[18]:


# Number of nulls

credit_data.isnull().sum()


# In[13]:


# Null % for all columns in data

null_perc = credit_data.isnull().sum()/len(credit_data)*100
null_perc


# In[14]:


# Top 60 cols with maximum null % 

null_perc.sort_values(ascending = False).head(60)


# Columns with a lot of NULLs are not useful for us as they would only capture data about a select applications. 
# 
# There is no standard rule for a good/bad % NULLs for columns to be used or discarded. It should be purely dependent on 
# use case and application of the EDA.
# 
# In our case, in order to keep this exercise simpler, we will discard all columns having more than 45% NULLs.

# ### 1.2 Data Cleaning 

# Identifying and removing columns with more than 45% nulls

# In[15]:


# Filtered list of Columns & NULL counts where NULL values are more than 45%
null_col = credit_data.isnull().sum().sort_values(ascending = False)
null_col = null_col[null_col.values >(0.45*len(credit_data))]
null_col


# In[24]:


no=len(null_col)
    print("There are "+ str(no) + " columns with more than 45% NULLs")


# Let's visually look at the columns with NULLs>45% and there NULL counts - 

# In[23]:


plt.figure(figsize=(20,4))
null_col.plot(kind='bar', color="red")
plt.title('Columns having more thn 45% nulls')
plt.show()


# ##### Removal of columns with NULLs>45%

# In[16]:


# Function to remove the columns having percentage of null values > 45%
def remove_null_cols(data):
    perc=0.45                                                             # Deciding cut-off NULL % to be 45%
    df = data.copy()                                                      # Creating a copy of data
    shape_before = df.shape                                               # Storing the shape of data before removal of columns
    remove_cols = (df.isnull().sum()/len(df))                             # Calculating % of NULLs
    remove_cols = list(remove_cols[remove_cols.values>=perc].index)       # Filtering cols with NULLs>45%
    df.drop(labels = remove_cols,axis =1,inplace=True)                    # Dropping cols
    print("Number of Columns dropped\t: ",len(remove_cols))    
    print("\nOld dataset rows,columns",shape_before,"\nNew dataset rows,columns",df.shape)
    return df


# In[17]:


# Removing cols with more than 45% nulls. Now we'll be using credit_data_1 for further analysis

credit_data_1 = remove_null_cols(credit_data)


# In[19]:


# Checking the % of null values for each column in new dataset
null_perc_1 = credit_data_1.isnull().sum()/len(credit_data_1)*100
null_perc_1.sort_values(ascending = False).head(60)


# We have now verified that our modified dataframe "Credit_data_1" has no cols with more than 31% NULLs

# In[20]:


credit_data_1["DAYS_LAST_PHONE_CHANGE"].isnull().sum()


# In[21]:


credit_data_1["OCCUPATION_TYPE"].unique()


# In[22]:


credit_data_1["OCCUPATION_TYPE"].isnull().sum()


# In[23]:


# How to calculate which value appearing how many times
# 
# 

# Values      | Count
# ------------------
# Laborers    | 10000
# Core staff  | 25000
# Accountants | 50000
# ..
# ..
# ..
# ..


# In[37]:


credit_data_1["OCCUPATION_TYPE"].value_counts()


# In[ ]:


credit_data_2 = credit_data_1.copy()


# In[40]:


# Fill null values of OCCUPATION_TYPE column using 'Laborers'
credit_data_2["OCCUPATION_TYPE"] = credit_data_1["OCCUPATION_TYPE"].fillna('Laborers')


# In[41]:


credit_data_2["OCCUPATION_TYPE"].isnull().sum()


# In[44]:


## Impute AMT_ANNUITY

credit_data_1["AMT_ANNUITY"].isnull().sum()


# ### 1.3 Imputing Missing Data

# The below listed columns can be categorized into a group of columns with similar significance as they all represent number of queries made to the Credit Bureau. 
# 
# Upon further investigation, we'll see that all have mode as 0. 
# We can impute NULLs for all of these with value 0. 
# 
# In the end it is also varified that there are 0 NULLs after imputation.

# ##### AMT_REQ_CREDIT_BUREAU_YEAR
# ##### AMT_REQ_CREDIT_BUREAU_MON
# ##### AMT_REQ_CREDIT_BUREAU_WEEK
# ##### AMT_REQ_CREDIT_BUREAU_DAY
# ##### AMT_REQ_CREDIT_BUREAU_HOUR
# ##### AMT_REQ_CREDIT_BUREAU_QRT

# In[31]:


credit_data_1


# In[24]:


# Checking value counts for AMT_REQ_CREDIT_BUREAU_YEAR
credit_data_1.AMT_REQ_CREDIT_BUREAU_YEAR.value_counts()

# We see that there are 71k 0s


# In[ ]:


# Similarly we can check for all. 
# Instead we can use the mode function to check modes for all of these variables

print(credit_data_1.AMT_REQ_CREDIT_BUREAU_YEAR.mode())
print(credit_data_1.AMT_REQ_CREDIT_BUREAU_MON.mode())
print(credit_data_1.AMT_REQ_CREDIT_BUREAU_WEEK.mode())
print(credit_data_1.AMT_REQ_CREDIT_BUREAU_DAY.mode())
print(credit_data_1.AMT_REQ_CREDIT_BUREAU_HOUR.mode())
print(credit_data_1.AMT_REQ_CREDIT_BUREAU_QRT.mode())


# In[25]:


credit_data_2 = credit_data_1.copy() # Making copy of our last data


# ##### Imputing NULLs with 0s

# In[26]:


# Imputing null with 0s

impute_list = ['AMT_REQ_CREDIT_BUREAU_YEAR','AMT_REQ_CREDIT_BUREAU_MON','AMT_REQ_CREDIT_BUREAU_WEEK',
               'AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_HOUR','AMT_REQ_CREDIT_BUREAU_QRT']

for i in impute_list:
        credit_data_2[i] = credit_data_1[i].fillna(0)
                         


# ##### Verifying count of NULLs after imputaion

# In[27]:


# Verifying count of NULLs after imputaion

print(credit_data_2['AMT_REQ_CREDIT_BUREAU_YEAR'].isnull().sum())
print(credit_data_2['AMT_REQ_CREDIT_BUREAU_MON'].isnull().sum())
print(credit_data_2['AMT_REQ_CREDIT_BUREAU_WEEK'].isnull().sum())
print(credit_data_2['AMT_REQ_CREDIT_BUREAU_DAY'].isnull().sum())
print(credit_data_2['AMT_REQ_CREDIT_BUREAU_HOUR'].isnull().sum())
print(credit_data_2['AMT_REQ_CREDIT_BUREAU_QRT'].isnull().sum())


# #### AMT_ANNUITY

# Since AMT_ANNUITY is a continuous variable, unlike AMT_REQ_CREDIT_BUREAU_YEAR etc ( which could take only integer values),
# it is better to impute this with the median value.
# 
# Another reason for chosing to go for Median instead of Mode is close value counts for top 2 values as we'll see below.

# In[ ]:


# Checking value counts for AMT_ANNUITY

credit_data_1.AMT_ANNUITY.value_counts()


# We see that top 2 value counts are close to each other. Thus we can chose to go for Median imputation.

# In[28]:


# median for AMT_ANNUITY

credit_data_1.AMT_ANNUITY.median()


# ##### Imputing NULLs with Median

# In[29]:


# Imputing NULLs with Median

credit_data_2['AMT_ANNUITY'] = credit_data_1['AMT_ANNUITY'].fillna(credit_data_1.AMT_ANNUITY.median())
credit_data_2['AMT_ANNUITY'].isnull().sum()


# #### AMT_GOODS_PRICE

# Similar to AMT_ANNUITY, imputing NULLs with Median for  AMT_GOODS_PRICE for similar reasons.

# In[ ]:


# Checking value counts for AMT_GOODS_PRICE

credit_data_1.AMT_GOODS_PRICE.value_counts() 


# In[30]:


# median for AMT_GOODS_PRICE

credit_data_1.AMT_GOODS_PRICE.median()


# In[31]:


# Imputing NULLs with Median

credit_data_2['AMT_GOODS_PRICE'] = credit_data_1['AMT_GOODS_PRICE'].fillna(credit_data_1.AMT_GOODS_PRICE.median())


# In[32]:


# Verifying count of NULLs to be 0

credit_data_2['AMT_GOODS_PRICE'].isnull().sum()


# ### 1.4 Fixing erroneous data

# As seen already with the help of describe function, we know that we need to treat -ve values in days columns.
# 
# We can modify the values to be absolute values, assuming that -ve sign was a technical fault during data feed. 

# In[ ]:


print(credit_data['DAYS_BIRTH'].unique())
print(credit_data['DAYS_EMPLOYED'].unique())
print(credit_data['DAYS_REGISTRATION'].unique())
print(credit_data['DAYS_ID_PUBLISH'].unique())
print(credit_data['DAYS_LAST_PHONE_CHANGE'].unique())


# In[34]:


credit_data['DAYS_BIRTH'].unique()


# In[35]:


for i in credit_data['DAYS_BIRTH'].unique():
    print(i)


# In[ ]:


# Confirming that all DAYS fields have -ve values

print(credit_data['DAYS_BIRTH'].unique())
print(credit_data['DAYS_EMPLOYED'].unique())
print(credit_data['DAYS_REGISTRATION'].unique())
print(credit_data['DAYS_ID_PUBLISH'].unique())
print(credit_data['DAYS_LAST_PHONE_CHANGE'].unique())


# In[ ]:


# Preparing the list of columns to be treated

erroneous_cols = [cols for cols in credit_data_2 if cols.startswith('DAYS')]
erroneous_cols


# In[ ]:


# Changing the column values with Absolute values using abs function

credit_data_2[erroneous_cols]= abs(credit_data_2[erroneous_cols])


# In[ ]:


# Verifying absence of -ve values in data

credit_data_2.describe()


# We have confirmed that there are no -ve values anymore.

# In[43]:


# Data in number of days
# Column YEARS_BIRTH - based of DAYS_BIRTH - calculate the number of years rounded to previous integer
# 53.4 => 53

credit_data_2["YEARS_BIRTH"] = credit_data_2["DAYS_BIRTH"] // 365
credit_data_2["YEARS_BIRTH"]


# In[ ]:


# Two columns

# YEARS_BIRTH, TARGET


# In[46]:


df = credit_data_2[['YEARS_BIRTH', 'TARGET']]


# In[48]:


df['YEARS_BIRTH'] = abs(df['YEARS_BIRTH'])


# In[49]:


df


# In[52]:


# Compare mean and median age of defaulters and non-defaulters
df.groupby('TARGET').agg({'YEARS_BIRTH': [np.mean, np.median]})


# In[55]:


# Replacing XNAs for CODE_GENDER


credit_data_2['CODE_GENDER'].value_counts()


# In[56]:


# Replace XNA with F in CODE_GENDER
credit_data_2['CODE_GENDER'] = credit_data_2['CODE_GENDER'].replace('XNA', 'F')


# In[57]:


credit_data_2['CODE_GENDER'].value_counts()


# In[58]:


# Who defaults more - M or F? based on credit_data_2

df = credit_data_2[['CODE_GENDER','TARGET']] 
df.value_counts()


# In[60]:


# F defaulters
14170/202452 * 100


# In[61]:


# M defaulters
10655/105059 * 100


# #### Replacing XNAs for CODE_GENDER

# A quick look on the value counts for CODE_GENDER shows 4 counts of XNA which is equivalent of a NULL.
# 
# Since 4 is a relatively small count, it doesn't matter much on how we impute it. Imputing it with Mode F in our case.

# In[ ]:


# CHecnking value counts for CODE_GENDER

credit_data_2.CODE_GENDER.value_counts()


# In[ ]:


# Replacing XNAs with F

credit_data_2.loc[credit_data_2.CODE_GENDER == 'XNA','CODE_GENDER'] = 'F'
credit_data_2.CODE_GENDER.value_counts()


# #### Replacing XNAs for ORGANIZATION_TYPE

# XNAs for ORGANIZATION_TYPE have 2nd highest count in the data. 
# We must be very careful in imputing such a high number of XNAs with any value.
# 
# Since it is a categorical variable, and there won't be any aggregrate functions performed on this data,
# we don't necessarily need whole of the value to be imputed.
# 
# Thus, changing all XNAs with NULLs to protect the originality of data.

# In[ ]:


# Checking value counts for ORGANIZATION_TYPE

credit_data_2.ORGANIZATION_TYPE.value_counts()


# In[ ]:


# Replacing XNAs with Nulls

credit_data_2['ORGANIZATION_TYPE'] = credit_data_1['ORGANIZATION_TYPE'].replace('XNA',np.NaN)


# In[ ]:


# Checking value counts for credit_data_2

credit_data_2.ORGANIZATION_TYPE.value_counts()


# We have confirmed that there are no more XNAs now in this field. 

# ### 1.5 Adding new columns by Binning Continuous Variables

# It is always a good practice to identify core or highly significant continuous fields in the data and then bin them into specific categories. It allows for an additional categorical analysis for such fields. We'll observe the use case of same later in this EDA exercise.
# For now, let's bin some of the continuous variables into 5 bins each as below -
# 

# In[62]:


credit_data_2


# In[65]:


help(pd.qcut)


# ##### Binning AMT_INCOME_TOTAL

# In[70]:


# Note : New function pd.qcut

# Using pd.qcut function to bin AMT_INCOME_TOTAL into 5 categories

credit_data_2['AMT_INCOME_RANGE'] = pd.qcut(credit_data_2.AMT_INCOME_TOTAL, 
                                            q=[0, 0.2, 0.5, 0.8, 0.95, 1], 
                                            labels=['VERY_LOW', 'LOW', "MEDIUM", 'HIGH', 'VERY_HIGH'])
credit_data_2['AMT_INCOME_RANGE'].head(7)


# In[ ]:


# Which INCOME_RANGE defaults the most?
# LOW - 8.6%




# In[1]:


# Which INCOME_RANGE takes the most number of loans?
df = credit_data_2[['AMT_INCOME_RANGE','TARGET']] 
df['AMT_INCOME_RANGE'].value_counts()['TARGET']


# # Which INCOME_RANGE has the highest mean value of amount of credit taken?
# 
# print(credit_data_2.groupby('AMT_INCOME_RANGE').mean()['AMT_CREDIT'])
# 
# 
# 
# 

# In[ ]:


# Scientific notation into number


# In[71]:


# Number of defaulters by INCOME_RANGE - absolute terms
credit_data_2.groupby('AMT_INCOME_RANGE').sum()['TARGET']


# In[75]:


df = credit_data_2[['AMT_INCOME_RANGE', 'TARGET']]


# In[78]:


df.value_counts()


# In[80]:


# Medium
8773/(97860 + 8773) * 100


# In[81]:


# Low
7755 / (7755+82334) * 100


# In[82]:


# Very Low
5225 / (5225 + 58446) * 100


# In[72]:


credit_data_2['AMT_INCOME_RANGE']


# In[74]:


credit_data_2.groupby('AMT_INCOME_RANGE').mean()['AMT_CREDIT']


# In[ ]:


df = credit_data_2[['AMT_INCOME_RANGE', ]]


# ##### Binning AMT_CREDIT

# In[ ]:


# Using pd.qcut function to bin AMT_CREDIT_RANGE into 5 categories

credit_data_2['AMT_CREDIT_RANGE'] = pd.qcut(credit_data_2.AMT_CREDIT, q=[0, 0.2, 0.5, 0.8, 0.95, 1], labels=['VERY_LOW', 'LOW', "MEDIUM", 'HIGH', 'VERY_HIGH'])
credit_data_2['AMT_CREDIT_RANGE'].head(7)


# ##### Binning DAYS_BIRTH

# In[ ]:


# Binning DAYS_BIRTH into 5 categories

# Step 1 : Convert days into Years ( Age)
# Step 2 : Bin as per Age of Applicant


credit_data_2['DAYS_BIRTH']= (credit_data_2['DAYS_BIRTH']/365).astype(int)
credit_data_2['DAYS_BIRTH'].unique()


# In[ ]:


# Using pd.qcut function to bin DAYS_BIRTH into 5 categories

credit_data_2['DAYS_BIRTH_BINS']=pd.cut(credit_data_2['DAYS_BIRTH'], 
                                      bins=[19,25,35,60,100], 
                                      labels=['Very_Young','Young', 'Middle_Age', 'Senior_Citizen'])


# In[ ]:


# Checking value counts for DAYS_BIRTH_BINS

credit_data_2['DAYS_BIRTH_BINS'].value_counts()


# ### 1.6 Splitting data based on TARGET

# Splitting data into 2 subsets based on Target Variable- Defaulter Data and Non-Defaulter Data.
# 
# This will help us with the comparison among 2 groups later 

# In[87]:


defaulter_df = credit_data_2[credit_data['TARGET']==1]
# Boolean Indexing 


# In[90]:


# defaulter_df


# In[88]:


non_defaulter_df = credit_data_2[credit_data['TARGET']==0]


# In[ ]:


# SPlitting data as per TARGET into deafulter and non-defaulter datasets

defaulter = credit_data_2[credit_data_2.TARGET==1]
non_defaulter =  credit_data_2[credit_data_2.TARGET==0]


# In[ ]:


# Checking row counts of data split as per TARGET

print(" Defaulter data shape - " + str(defaulter.shape) )
print(" Non-Defaulter data shape - " + str(non_defaulter.shape) )


# In[ ]:


# Checking % of data split as per TARGET

print(" Defaulter data % - " + str(round(defaulter.shape[0]*100/credit_data_2.shape[0],2) ))
print(" Non-Defaulter data % - " + str(round(non_defaulter.shape[0]*100/credit_data_2.shape[0],2) ))


# ## 2. Univariate Analysis

# Univariate Analysis is simplest form of analysing data. It restricts the analysis to only 1 variable as the name states.
# ( Uni means One )
# 
# It doesn't take into account the mutual relationships and associations among variables. Rather it focuses on 
# finding patterns through a particular field. 

# ##### Occupation Type

# In[ ]:


# Distribution of 'OCCUPATION_TYPE'

temp = credit_data_2["OCCUPATION_TYPE"].value_counts()                      # Value counts for Occupation type
sns.barplot(x=temp.index, y = temp.values, color = 'maroon')                # Plotting bar graph
_=plt.xticks(rotation=90, size = 14)                                        # Rotating x axis ticks so that values dont overlap
_=plt.yticks( size = 14)                                                    # Adjusting size for y txis ticks
_=plt.title('Loan Applications by Occupation Type', size=18,color = 'maroon') # Chart title


# We can infer that most of the applications come for Labourers, Sales Staff and Core Staff. 

# ##### Organization Type

# In[ ]:


# Distribution of 'Organization Type'
plt.figure(figsize=(20,4))
temp = credit_data_2["ORGANIZATION_TYPE"].value_counts()
sns.barplot(x=temp.index, y = temp.values, color = 'maroon')
_=plt.xticks(rotation=90, size = 14)
_=plt.yticks( size = 14)
_=plt.title('Loan Applications by Organization Type', size=18,color = 'maroon')


# It is observed that majority of the applicants belong to Business Entity Type 3 an Self Employed. 

# ##### Comparison of Gender Distribution among Defaulters and Non Defaulters

# In[ ]:


## Code to See the Comparison of Gender Applicants Distribution among Defaulters and Non-Defaulters 

colors = sns.color_palette('tab10')[0:5]                   # Setting Color pallette for pie chart

fig, axes=plt.subplots(nrows =1,ncols=2,figsize=(20,12))   # Defining Subplots and figure size. Keeping it wider for 2 chart
axes[0].set_title("Box Plot of  " )                        # Setting title for Subplot 1
data = non_defaulter['CODE_GENDER'].value_counts()         # Data prep fot Subplot 1 ( Non Defaulter )
data_df = pd.DataFrame({'labels': data.index,'values': data.values})     

# Pie chart for Subplot 1 ( Non Defaulter part )
_=axes[0].pie(data_df['values'], labels = data_df['labels'], colors = colors, autopct='%0.1f%%',textprops={'fontsize': 14})
_=axes[0].set_title('Applicants by  '+'CODE_GENDER', size=18,color = '#291038')
_=axes[0].legend()

# Pie chart for Subplot 2 ( Defaulter part )
axes[1].set_title("Box Plot of  " )
data = defaulter['CODE_GENDER'].value_counts()
data_df = pd.DataFrame({'labels': data.index,'values': data.values})
_=axes[1].pie(data_df['values'], labels = data_df['labels'], colors = colors, autopct='%0.1f%%',textprops={'fontsize': 14})
_=axes[1].set_title('Defaulters by  '+'CODE_GENDER', size=18,color = '#291038')
_=axes[1].legend()


# Insights - 
# 
# * There is majority of Male loan apllicants
# * More Men deafult loans as compared to Women, since the % split has increased further for Men in case of Defaulter distribution.

# Often we would require to re-use the same code for multiple combination of variables.
# 
# It is a common practice to prepare charts by calling functions instead of re-writing the code again and again.
# 
# There are following benefits to this appraoch - 
# * Code Modularity is improved
# * Less code is required to perform same amount of task
# * Notebook looks more cleaner

# ##### Converting above code to a function - 

# In[ ]:


# Function for Univariate Comarison

def univariate_comparison(col,hue=None):
    colors = sns.color_palette('tab10')[0:5]

    fig, axes=plt.subplots(nrows =1,ncols=2,figsize=(20,12))
    axes[0].set_title("Box Plot of  " )
    data = non_defaulter[col].value_counts()
    data_df = pd.DataFrame({'labels': data.index,'values': data.values})
    _=axes[0].pie(data_df['values'], labels = data_df['labels'], colors = colors, autopct='%0.1f%%',textprops={'fontsize': 14})
    _=axes[0].set_title('Applicants by  '+col, size=18,color = '#291038')
    _=axes[0].legend()

    axes[1].set_title("Box Plot of  " )
    data = defaulter[col].value_counts()
    data_df = pd.DataFrame({'labels': data.index,'values': data.values})
    _=axes[1].pie(data_df['values'], labels = data_df['labels'], colors = colors, autopct='%0.1f%%',textprops={'fontsize': 14})
    _=axes[1].set_title('Defaulters by  '+col, size=18,color = '#291038')
    _=axes[1].legend()


# ##### Comparison of Income Type Distribution among Defaulters and Non Defaulters

# In[ ]:


# Comparison of Income Type Distribution among Defaulters and Non Defaulters

univariate_comparison('NAME_INCOME_TYPE')


# Insights - 
# * Almost half of the Loan applications come from Working professionals.
# * Working professionals contribute more than expected to loan defaults. The % split has increased from 51% to 61%

# ##### Comparison of Family Status Distribution among Defaulters and Non Defaulters

# In[ ]:


# Comparison of Family Status Distribution among Defaulters and Non Defaulters

univariate_comparison('NAME_FAMILY_STATUS')


# Insights-
# * 65 % of the Loan applicants are married.
# * Family status doesn't seem to have any major impact on Loan deafults.

# ##### Comparison of Education Type Distribution among Defaulters and Non Defaulters

# In[ ]:


# Comparison of Education Type Distribution among Defaulters and Non Defaulters

univariate_comparison('NAME_EDUCATION_TYPE')


# Insights-
# * More than 2/3rds of Loan applicants have highest education as Secondary. 
# * Secondary Education class contribute majorly ( more than expected too) for loan defaults.
# * There is a considerable decrease in % split for loan defaults by people with higher education. ( from 25% to 16%)

# ##### Comparison of Housing Type Distribution among Defaulters and Non Defaulters

# In[ ]:


# Comparison of Housig Type Distribution among Defaulters and Non Defaulters


univariate_comparison('NAME_HOUSING_TYPE')


# Insights-
# * Almost 90% of Loan applicants have their own home.
# * Housing type doesn't play a significant role in determining whether there will be a loan defaulter. 

# ##### Comparison of Income Range Distribution among Defaulters and Non Defaulters

# In[ ]:


# Comparison of Income Range Distribution among Defaulters and Non Defaulters

univariate_comparison('AMT_INCOME_RANGE')


# Insights-
# * Here also, the % split is more or less unchanged for Defaulters. It suggests that Income doesn't play a significant role in loan defaults. Although, further drilldown analysis ( later done in this notebook ) would tell us a different story.
# 
# It is always good practice to verify our hypotheses by multiple checks and not jump onto conclusions quickly. 

# NOTE : Let's recall that AMT_INCOME_RANGE is a derived variable created by binning earlier. 
# This how binning can be useful in EDA, while this is just one use case, it has many other applications in ML as well. 

# ##### Comparison of Age Distribution among Defaulters and Non Defaulters

# In[ ]:


# Comparison of Age Distribution among Defaulters and Non Defaulters

univariate_comparison('DAYS_BIRTH_BINS')


# Insights - 
# * There is a significant shift in % split for Middle Age and Young applicants.
# * Middle Aged applicants are contributing lesser to loan defaults
# * Young applicants are more expected to default on a loan since there is a change in % aplit from 24% to 32%

# ##### Comparison of Loan Type Distribution among Defaulters and Non Defaulters

# In[ ]:


# Comparison of Loan Type Distribution among Defaulters and Non Defaulters

univariate_comparison('NAME_CONTRACT_TYPE')


# Insights-
# * Cash loans are slightly more likely to be defaulted than revolving loans. 

# ##### Comparison of Accompany Type Distribution among Defaulters and Non Defaulters

# In[ ]:


# Comparison of Accompany Type Distribution among Defaulters and Non Defaulters

univariate_comparison('NAME_TYPE_SUITE')


#  Insights-
# * Majority of loans are applied by single occupants
# * This parameter doesn't have any impact on loan defaults as the % split is unchanged in both cases.

# ### Univariate Analysis of Quantitative Variables

# In[ ]:


# Defining function for Univariate Analysis of Quantitative Variables

def univariate_comparison_quant(col,hue=None):

    fig, axes=plt.subplots(nrows =2,ncols=2,figsize=(20,12))                      # Defining 4 subplots, changing fig size
    axes[0,0].set_title("Displot (Non-Defaulter) for  " + col )                   # Chart titl for Subplot 1
    sns.distplot(non_defaulter[~non_defaulter[col].isna()][col],ax=axes[0,0], color="#4CB391") # Distplot in subplot 1


    axes[0,1].set_title("Displot (Defaulter) for  " + col )                       #  Title for Subplot 2
    sns.distplot(defaulter[~defaulter[col].isna()][col],ax=axes[0,1], color="#4CB391") # Displot in Subplot 2
    
    axes[1,0].set_title("Boxplot (Non-Defaulter) for  " + col )                   # Title for Subplot 3
    sns.boxplot(non_defaulter[~non_defaulter[col].isna()][col],ax=axes[1,0], color="#4CB391") # Boxplot in subplot 3

    axes[1,1].set_title("Boxplot (Defaulter) for  " + col )                       # Title for Subplot 4
    sns.boxplot(defaulter[~defaulter[col].isna()][col],ax=axes[1,1], orient='h',color="#4CB391") # Boxplot in Subplot 4

    plt.tight_layout()


# In[ ]:


# Univariate Analysis for Annuity Amount

univariate_comparison_quant('AMT_ANNUITY')


# Insights - 
# * Applicants with lower Annuity Amount are slightly more likely to default on a loan.
# * Majority of Loan applicants come from 1st quartile of Annuity data ( Low salary people )

# In[ ]:


# Univariate Analysis for Loan Amount

univariate_comparison_quant('AMT_CREDIT')


# Insights-
# * Loan Amount doesn't seem to have any correlation with Loan defaults. 

# In[ ]:


# Univariate Analysis for Goods Price Amount

univariate_comparison_quant(col='AMT_GOODS_PRICE')


# Insights-
# * The distribution are almost unchanged for Defaulters and Non Defaulters, hence we can say that Goods Price doesn't impact the chance of a loan default.

# ## 3. Bivariate & Multivariate Analysis

# Bivariate Analysis - 
# 
# It is one of the simplest form of statistical analysis where 2 variables are involved. It looks for relationship among the 2 variables.
# The applications involve hypothesis validation of association among variables, finding trends, regression etc.
# 
# Multivariate Analysis-
# 
# When more than 2 variable are involved in an analysis, it will be a multi-variate analysis. The additional variables may take form of hue color, 3rd axis etc. 

# In[ ]:


# Function for Multivariate analysis

def multivariate(col1,col2,col3):                                            # Takes 3 columns as inputs

    fig, axes=plt.subplots(nrows =1,ncols=2,figsize=(20,12))               
    
    axes[0].set_title("Boxplot (Non-Defaulter) for  "  )
    _=sns.boxplot(data=non_defaulter,x=col1, y=col2,palette = 'rainbow', hue= col3,ax=axes[0])
    _=axes[0].set_title('Loan Amount by  ' + col2 + ' & ' + col3 + ' (Non-Defaulter)', size=15,color = 'blue')

    axes[1].set_title("Boxplot (Defaulter) for  "  )
    _=sns.boxplot(data=defaulter,x=col1, y=col2,palette = 'rainbow', hue= col3,ax=axes[1])
    _=axes[1].set_title('Loan Amount by  ' + col2 + ' & ' + col3 + ' (Defaulter)', size=15,color = 'blue')


# In[ ]:


# Analysis of AMT_INCOME_RANGE, AMT_CREDIT & NAME_FAMILY_STATUS

multivariate('AMT_INCOME_RANGE','AMT_CREDIT','NAME_FAMILY_STATUS')


# Insights-
# * With increase in Income range, the loan amount increases proportionally.
# * On family status axis, we observe that Married applicants have higher loan amount than others.
# 

# In[ ]:


# Analysis of NAME_EDUCATION_TYPE, AMT_CREDIT & NAME_FAMILY_STATUS

multivariate('NAME_EDUCATION_TYPE','AMT_CREDIT','NAME_FAMILY_STATUS')


# Insights-
# * Higher the education, lesser is the likelihood of a loan default
# * Among different family status, married ones have the highest likelihood of loan default

# #### Drilldown Analysis

# Here we'll look for % defaulters within different classes in a particular variable. 

# In[ ]:


# Defining function for drilldown analysis

def perc_defaulters(col):

    fig, axes=plt.subplots(nrows =1,ncols=2,figsize=(20,10))
    
    total = credit_data_2[[col,'TARGET']].groupby(col).count()
    defaulter_1 = defaulter[[col,'TARGET']].groupby(col).count()
    perc = defaulter_1*100/total
    
    axes[0].set_title("Application Counts by  "+ col  )
    _=sns.barplot(x=total.index,y=total.TARGET,color='grey',order=total.sort_values('TARGET',ascending=False).index,ax=axes[0])
    _=axes[0].set_xticklabels(total.sort_values('TARGET',ascending=False).index,rotation=60, ha='right')

    axes[1].set_title("Defaulter % by " + col  )
    _=sns.barplot(x=perc.index,y=perc.TARGET,color='#ff597d',order=perc.sort_values('TARGET',ascending=False).index,ax=axes[1])
    _=axes[1].set_xticklabels(perc.sort_values('TARGET',ascending=False).index,rotation=60, ha='right')


# In[ ]:


# Drilldown analysis of AMT_INCOME_RANGE

perc_defaulters('AMT_INCOME_RANGE')


# Insights-
# * Median income range professionals have maximum applications in the data
# * Low Income range have maximum % of loan defaults
# * As the Income range increases, loan default probability decreases

# In[ ]:


# Drilldown analysis of NAME_INCOME_TYPE

perc_defaulters('NAME_INCOME_TYPE')


# Insights-
# * Applicants on Maternity leave have a whopping 40% loan default rate
# * The second to the list are Unemployed applicants with 35% loan defaults

# In[ ]:


# Drilldown analysis of NAME_CONTRACT_TYPE
perc_defaulters('NAME_CONTRACT_TYPE')


# Insights-
# * Majority of the loans are cash loans. Cash loans also have almost double probability of a loan default than revolving loans.

# In[ ]:


# Drilldown analysis of NAME_EDUCATION_TYPE
perc_defaulters('NAME_EDUCATION_TYPE')


# Insights-
# * Higher the education of an applicant, lesser the chance of loan default
# * Lower secondary applicants have a concerning 11% loan default rate, but the count of applicants is low 
# * The major concern is of Secondary education applicants. They have highest applicants and a significant 9% loan default rate as well. 

# In[ ]:


# Drilldown analysis of OCCUPATION_TYPE

perc_defaulters('OCCUPATION_TYPE')


# Insights-
# * Low skill labourers  have an alarming 17% loan default rate. The positive here is that they don't have a high applicant count.
# * Labourers  & Sales staff will be a major area of concern here, with maximum applicants and a significant loan default rate as well. 
# * Drivers also have an alarming combination of counts and default %.

# ##### Pivot table of all loan default %

# In[ ]:


perc_defaulters= pd.pivot_table(credit_data_2, values='TARGET', 
                      index=['CODE_GENDER','AMT_INCOME_RANGE'],
                      columns=['NAME_EDUCATION_TYPE'], aggfunc=np.mean)
perc_defaulters*100


# Insights - 
# 
# Categories with more than 9% default rate - 
# * Females, High Income, Academic degree
# * Male, Very Low income , Incomplete higher
# * Male, Low Income , Incomplete higher
# * Male, Medium Income , Incomplete higher
# * Female, Low Income, Lower Secondary
# * Female, Medium Income, Lower Secondary
# * Male, Very Low Income, Lower Secondary
# * Male, Low Income, Lower Secondary
# * Male, Medium Income, Lower Secondary
# * Male, {ALL INCOME RANGES} , Secondary
# 

# ##### Bivariate Analysis using Pairplot 

# In[ ]:


# Data for Pairplot

pairplor_data = credit_data_2[['AMT_CREDIT', 'AMT_ANNUITY', 'AMT_INCOME_TOTAL', 'AMT_GOODS_PRICE', 'DAYS_BIRTH','TARGET']].fillna(0)


# In[ ]:


# Plotting pairplot
_=sns.pairplot(pairplor_data,hue='TARGET',diag_kind='kde')


# Insights-
# 
# * AMT_CREDIT & AMT_GOODS_PRICE are correlated  ( With higher priced goods, loan amount is higher)
# * AMT_ANNUITY & AMT_GOODS_PRICE are also correlated ( With higher annuity, expensive goods are purchased)
# * AMT_ANNUITY & AMT_CREDIT are correlated (Higher the annuity,higher the loan amount)
# 
# With respect to TARGET - 
# * Loan defaulters ( Blue ) are younger in age
# 

# ##### Correlation Check using Heatmap

# In[ ]:


# Data prep for heatmap
heatmap_data = credit_data_2[['AMT_GOODS_PRICE','AMT_INCOME_TOTAL','AMT_ANNUITY','DAYS_EMPLOYED','DAYS_BIRTH',
         'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH',
         'AMT_CREDIT',]].corr(method = 'pearson')


# In[ ]:


# Plotting heatmap
_=sns.heatmap(heatmap_data, cmap='YlGnBu')


# Insights-
# * The heatmap confirms our correlation findings from pariplot

# ##### Top 10 correlations in the data

# In[ ]:


# Preparing data for getting top 10 correlation combinations 

corr_matrix=defaulter[['AMT_GOODS_PRICE','AMT_INCOME_TOTAL','AMT_ANNUITY','DAYS_EMPLOYED',
  'DAYS_BIRTH', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH',
  'AMT_CREDIT']].corr(method = 'pearson')                     # Getting Correaltion Matrix

# Filtering top half traingle usng np.triu
corr_matrix=corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(np.bool))

corr_matrix_df=corr_matrix.unstack().reset_index()            # Unstacking the last dataframe to get combos of 2 variables 
corr_matrix_df


# In[ ]:


corr_matrix_df.columns = ['Variable 1','Variable 2','Correlation']          # Naming the cols apprpriately
corr_matrix_df.dropna(subset=['Correlation'],inplace=True)                  # Dropping NAs

# Adding absolute column as we are interested in magnitude
corr_matrix_df['Correlation ( Absolute )']=corr_matrix_df['Correlation'].abs() 

# Sorting by top correlations and getting top 10 combos
corr_matrix_df.sort_values('Correlation ( Absolute )', ascending=False).head(10)


# # 4. Final Insights 

# Following are the driving factors for a loan default - 
# 
# * Lower the highest education of an applicant, higher the chance of loan default. 
# This is one of the core driving factor in loan defaults.
# 
# * Labourers & Sales staff are major area of concern , with maximum applicants and a significant loan default rate. Drivers also have an alarming combination of counts and default %.
# 
# * Applicants on Maternity leave have a whopping 40% loan default rate. Unemployed applicants also have 35% loan defaults
# 
# * Low Income range have maximum % of loan defaults. As the Income range increases, loan default probability decreases
# 
# * Among different family status, married ones have the highest likelihood of loan default
# 
# * Applicants with lower Annuity Amount are slightly more likely to default on a loan.
# 
# * Young applicants are more expected to default on a loan.
# 
# * More Men deafault loans as compared to Women
# 
