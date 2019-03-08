# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 17:04:47 2019

@author: lucas
"""

#importing the necessary packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#displaying all the columns when called
pd.set_option('display.max_columns', 500)

df = pd.read_excel('birthweight.xlsx')

#Dataset exploration

df.columns


df.shape


df.describe().round(2)


df.info()

##################################################################
#flagging missing values
##################################################################

df.isnull().sum()

for col in df:

    """ Create columns that are 0s if a value was not missing and 1 if
    a value is missing. """
    
    if df[col].isnull().any():
        df['m_'+col] = df[col].isnull().astype(int)
        
df_dropped = df.dropna()
        

#checking missing values from number of prenatal visits

sns.distplot(df_dropped['npvis'])

df['npvis'].describe()

### npvis is relatively noral, filling with the median
fill = df['npvis'].median()

df['npvis'] = df['npvis'].fillna(fill)


#checking missing values from father's education


df['feduc'].describe()

sns.distplot(df_dropped['feduc'])


### filling with the median 

fill = df['feduc'].median()

df['feduc'] = df['feduc'].fillna(fill)


#checking missing values for father's age


df['fage'].describe()

sns.distplot(df_dropped['fage'])


### filling with the median 

fill = df['fage'].median()

df['fage'] = df['fage'].fillna(fill)


#checking missing values for avg drinks per week

df['drink'].describe()


#analyzing the na of column drink values' relationship with the baby weight


df_drinks = df.loc[: , 'drink']

df_drinks = 





