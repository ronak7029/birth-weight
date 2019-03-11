#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 09:25:26 2019

@author: danielasantacruzaguilera
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf 
from sklearn.model_selection import train_test_split # train/test split
from sklearn.neighbors import KNeighborsRegressor
import sklearn.metrics
from sklearn.model_selection import cross_val_score

excel= 'birthweight.xlsx'

birth= pd.read_excel(excel)

print(birth.info())
print(birth.shape)
print(birth.isnull().sum().sum())

#Missing values in the columns:
print(birth.isnull().any())


#Number of  missing values we have per column:
print(birth[:].isnull().sum())

#Flagging missing values: 
for col in birth:
    if birth[col].isnull().astype(int).sum() > 0:
        birth['m_'+col] = birth[col].isnull().astype(int)
        
df_b= pd.DataFrame.copy(birth)


#Correlation matrix and heatmap
correlation = df_b.iloc[:, :-11].corr()
sns.heatmap(correlation, 
            xticklabels=correlation.columns.values,
            yticklabels=correlation.columns.values)
plt.savefig('birthWeight.png')
plt.show()
#There is no correlation between birth weight and any other variable

#Histograms
fillna= pd.DataFrame.copy(df_b)
fillna= fillna.fillna(0)

for col in fillna.iloc[:, :-11]:
    sns.distplot(fillna[col])
    plt.tight_layout()
    plt.show()


#Boxplots
for col in fillna.iloc[:, :-11]:
   fillna.boxplot(column = col, vert = False)
   plt.title(f"{col}")
   plt.tight_layout()
   plt.show()
   
#Counting non zero values per column
fillna.astype(bool).sum(axis=0)
# cigarretes has 147 nonzero values 
#drink has 16 nonzero values
#the majority of race is white 1630/1832 for father, 1624/1832 for mother

#############################################################################
#1575 women did not smoke along with 110 missing values for smoking. 
#This means 1685/1872 observations do not involve mother smoking. 
#This results in a low correlation between our birthweight and smoking
# because there is not enough data on mothers who smoke to be significant. 
#We face a similar situation with our Alcohol Variable. 
#We find 1701 mothers who do not drink alcohol and we find another 115 
#observations have missing values for alcohol. Therefore 1816/1872 
#observations do not have any impact upon low birthweight. 
#The lack of data on mothers who drank alcohol is also leading us to find 
#little correlation with birthweight.  
##############################################################################

#ANALYZING THE NEW DATASET FOR LOW BIRTHWEIGHT
birthfeature= 'birthweight_feature_set.xlsx' 

df= pd.read_excel(birthfeature)

# Column names
df.columns


# Displaying the first rows of the DataFrame
print(df.head())


# Dimensions of the DataFrame
df.shape


# Information about each variable
df.info()


# Descriptive statistics
df.describe().round(2)

#Total Number of Misisng values
print(df.isnull().sum().sum())

#Missing values in the columns:
print(df.isnull().any())

#Number of  missing values we have per column:
print(df[:].isnull().sum())

#Flagging missing values: 
for col in df:
    if df[col].isnull().astype(int).sum() > 0:
        df['m_'+col] = df[col].isnull().astype(int)
      
#Checking if the for loop worked: 
print(df.head())
A = df.isnull().sum().sum()
B = df.iloc[:,-18:].sum().sum()

if A == B:
    print('All missing values accounted for.')
else:
    print('Some missing values may be unaccounted for, please audit.')
    
#imputing missing values

df_dropped = pd.DataFrame.copy(df)
df_dropped= df_dropped.dropna()

sns.distplot(df_dropped['meduc'])
sns.distplot(df_dropped['npvis'])
sns.distplot(df_dropped['feduc'])

#Filling with median
fill = df['meduc'].median()

df['meduc'] = df['meduc'].fillna(fill)


fill = df['npvis'].median()

df['npvis'] = df['npvis'].fillna(fill)


fill = df['feduc'].median()

df['feduc'] = df['feduc'].fillna(fill)

#checking for more missing values:
print(df.isnull().any().any())

###############################################################################
# Outlier Analysis
###############################################################################

df_quantiles = df.loc[:, :].quantile([0.20,
                                                0.40,
                                                0.60,
                                                0.80,
                                                1.00])

    
print(df_quantiles)



for col in df:
    print(col)

"""
Assumed Continuous/Interval Variables:
mage
monpre
npvis
fage
omaps
fmaps
cigs
drink
bwght


Assumed Ordinal:
meduc
feduc


Binary Classifiers:
male
mwhte
mblck
moth
fwhte
fblck
foth

"""

##########################################
######EDA: Histograms
for col in df.iloc[:, :17]:
    sns.distplot(df[col], bins = 'fd')
    plt.tight_layout()
    plt.show()

#####Boxplots
    
df.boxplot(column = ['male',
'mwhte',
'mblck',
'moth',
'fwhte',
'fblck',
'foth'],
                vert = False,
                 manage_xticks = True,
                 patch_artist = False,
                 meanline = True,
                 showmeans = True,)
plt.savefig('binarybox.png')


df.boxplot(column = ['mage',
'fage'],
                vert = False,
                 manage_xticks = True,
                 patch_artist = False,
                 meanline = True,
                 showmeans = True,)
plt.savefig('birthboxplots1.png')

df.boxplot(column = ['meduc','feduc'],
                vert = False,
                 manage_xticks = True,
                 patch_artist = False,
                 meanline = True,
                 showmeans = True,)
plt.savefig('birthboxplots2.png')

df.boxplot(column = ['drink'],
                vert = False,
                 manage_xticks = True,
                 patch_artist = False,
                 meanline = True,
                 showmeans = True,)
plt.savefig('birthboxplots3.png')

df.boxplot(column = ['cigs'],
                vert = False,
                 manage_xticks = True,
                 patch_artist = False,
                 meanline = True,
                 showmeans = True,)
plt.savefig('birthboxplots4.png')

df.boxplot(column = ['omaps'],
                vert = False,
                 manage_xticks = True,
                 patch_artist = False,
                 meanline = True,
                 showmeans = True,)
plt.savefig('birthboxplots5.png')

df.boxplot(column = ['fmaps'],
                vert = False,
                 manage_xticks = True,
                 patch_artist = False,
                 meanline = True,
                 showmeans = True,)
plt.savefig('birthboxplots8.png')

df.boxplot(column = ['npvis'],
                vert = False,
                 manage_xticks = True,
                 patch_artist = False,
                 meanline = True,
                 showmeans = True,)
plt.savefig('birthboxplots6.png')

df.boxplot(column = ['monpre'],
                vert = False,
                 manage_xticks = True,
                 patch_artist = False,
                 meanline = True,
                 showmeans = True,)
plt.savefig('birthboxplots7.png')

df.boxplot(column = ['bwght'],
                vert = False,
                 manage_xticks = True,
                 patch_artist = False,
                 meanline = True,
                 showmeans = True,)
plt.savefig('weightbox.png')
    
########################
# Tuning and Flagging Outliers
########################

df_quantiles = df.loc[:, :].quantile([0.05,
                                                0.40,
                                                0.60,
                                                0.80,
                                                0.95])
    
Q1 = df.loc[:, :].quantile(0.25)
Q3 = df.loc[:, :].quantile(0.75)
IQR = Q3 - Q1
high = Q3 + 1.5 * IQR
low = Q1 - 1.5 * IQR
print(high.round())
print(low.round())
    

#Outlier Flags
mage_high= 66
monpre_high=4
npvis_low=7
npvis_high=15
fage_high= 55
feduc_low=6
omaps_low=6
fmaps_low=9
fmaps_high=9
drink_high=12

#######mage
df['out_mage'] = 0

for val in enumerate(df.loc[ : , 'mage']):
    
    if val[1] >= mage_high:
        df.loc[val[0], 'out_mage'] = 1
        
#####monpre
df['out_monpre'] = 0

for val in enumerate(df.loc[ : , 'monpre']):
    
    if val[1] >= monpre_high:
        df.loc[val[0], 'out_monpre'] = 1

#####npvis
df['out_npvis'] = 0

for val in enumerate(df.loc[ : , 'npvis']):
    
    if val[1] >= npvis_high:
        df.loc[val[0], 'out_npvis'] = 1


for val in enumerate(df.loc[ : , 'npvis']):
    
    if val[1] <= npvis_low:
        df.loc[val[0], 'out_npvis'] = 1

####fage
df['out_fage'] = 0

for val in enumerate(df.loc[ : , 'fage']):
    
    if val[1] >= fage_high:
        df.loc[val[0], 'out_fage'] = 1  
        
####feduc
df['out_feduc'] = 0

for val in enumerate(df.loc[ : , 'feduc']):
    
    if val[1] <= feduc_low:
        df.loc[val[0], 'out_feduc'] = 1  

#####omaps
df['out_omaps'] = 0

for val in enumerate(df.loc[ : , 'omaps']):
    
    if val[1] <= omaps_low:
        df.loc[val[0], 'out_omaps'] = 1  

#####fmaps
df['out_fmaps'] = 0

for val in enumerate(df.loc[ : , 'fmaps']):
    
    if val[1] > fmaps_high:
        df.loc[val[0], 'out_fmaps'] = 1


for val in enumerate(df.loc[ : , 'fmaps']):
    
    if val[1] < fmaps_low:
        df.loc[val[0], 'out_fmaps'] = 1

####drink
df['out_drink'] = 0

for val in enumerate(df.loc[ : , 'drink']):
    
    if val[1] >= drink_high:
        df.loc[val[0], 'out_drink'] = 1
    

###############################################################################
# Correlation Analysis
###############################################################################
df.head()
df_corr = df.iloc[:, :18].corr().round(2)

print(df_corr)

df_corr.loc['bwght'].sort_values(ascending = False)
  
#Correlation matrix and heatmap
correlation2 = df.iloc[:, :-11].corr()
sns.heatmap(correlation2, 
            xticklabels=correlation2.columns.values,
            yticklabels=correlation2.columns.values)
plt.savefig('birthWeight2.png')
plt.show()

###############################################################################
# Univariate Regression Analysis
###############################################################################

########################
# Full Model
########################

for col in df:
    print(col)

lm_full = smf.ols(formula = """bwght ~ mage + 
                              meduc +
                              monpre +
                              npvis +
                              fage +
                              feduc +
                              omaps +
                              fmaps +
                              cigs +
                              drink +
                              male +
                              mwhte +
                              mblck +
                              moth +
                              fwhte +
                              fblck +
                              foth +
                              m_meduc +
                              m_npvis +
                              m_feduc +
                              out_mage +
                              out_monpre +
                              out_npvis +
                              out_fage +
                              out_feduc +
                              out_omaps +
                              out_fmaps +
                              out_drink
                                           """,
                         data = df)


# Fitting Results
results = lm_full.fit()



# Printing Summary Statistics
print(results.summary())

print(f"""
Summary Statistics:
R-Squared:          {results.rsquared.round(3)}
Adjusted R-Squared: {results.rsquared_adj.round(3)}
""")
    
    
"""
    The model accounts for 74.0% of the variance but some of the variables have unacceptable p-values.
    Let's consider removing these variables.
"""

########################
# Significant Model
########################

lm_sig = smf.ols(formula = """bwght ~ mage + 
                              cigs +
                              drink +
                              mwhte +
                              mblck +
                              moth +
                              fwhte +
                              fblck +
                              foth
                                           """,
                         data = df)


# Fitting Results
results = lm_sig.fit()



# Printing Summary Statistics
print(results.summary())

print(f"""
Summary Statistics:
R-Squared:          {results.rsquared.round(3)}
Adjusted R-Squared: {results.rsquared_adj.round(3)}
""")
   
    
    
lm_1 = smf.ols(formula = """bwght ~ mage + 
                              cigs +
                              drink +
                              mwhte+
                              mblck+
                              moth                             
                                     """,
                         data = df)


# Fitting Results
results = lm_1.fit()



# Printing Summary Statistics
print(results.summary())

print(f"""
Summary Statistics:
R-Squared:          {results.rsquared.round(3)}
Adjusted R-Squared: {results.rsquared_adj.round(3)}
""")
    
    
################Recoding Race

df['race']= 0

for val in enumerate(df.loc[ : , 'mwhte']):
    
    if val[1] == 1:
        df.loc[val[0], 'race'] = 0
        
for val in enumerate(df.loc[ : , 'mblck']):
    
    if val[1] == 1:
        df.loc[val[0], 'race'] = 1
    
###############################################################################
# Generalization using Train/Test Split
###############################################################################

df_data= df.drop(['bwght','m_meduc','m_npvis','m_feduc','out_mage','out_monpre',
           'out_npvis','out_fage','out_feduc','out_omaps','out_fmaps',
           'out_drink','race'],axis = 1)



df_target = df.loc[:, 'bwght']



X_train, X_test, y_train, y_test = train_test_split(
                                               df_data,
                                               df_target, test_size = 0.10,
                                               random_state = 508)

# Let's check to make sure our shapes line up.

# Training set 
print(X_train.shape)
print(y_train.shape)

# Testing set
print(X_test.shape)
print(y_test.shape)

