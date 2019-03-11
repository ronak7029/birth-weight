# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 04:32:15 2019

@author: bhagyashree
"""

#importing the necessary packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#displaying all the columns when called
file = 'birthweight_feature_set.xlsx'

bwt_df = pd.read_excel(file)


#Dataset exploration

bwt_df.columns

bwt_df.shape

bwt_df.describe().round(2)

bwt_df.info()

bwt_df.corr()

################################################################
#Flagging missing values 
################################################################

bwt_df.isnull().sum()

for col in bwt_df:

    """ Create columns that are 0s if a value was not missing and 1 if
    a value is missing. """
    
    if bwt_df[col].isnull().any():
        bwt_df['m_'+col] = bwt_df[col].isnull().astype(int)
        
################################################################
#checking missing values from omaps, fmaps &  cigs
################################################################



bwt_df['omaps'].describe()

bwt_df['fmaps'].describe()

bwt_df['cigs'].describe()


################################################################
#General Outlier Analysis
################################################################

plt.subplot(2, 2, 1)
sns.distplot(bwt_df['omaps'],
             bins = 'fd',
             color = 'y')

plt.xlabel('1 min apgar score')


########################


plt.subplot(2, 2, 2)
sns.distplot(bwt_df['fmaps'],
             bins = 'fd',
             color = 'g')

plt.xlabel('5 min apgar score')



########################


plt.subplot(2, 2, 3)
sns.distplot(bwt_df['cigs'],
             bins = 'fd',
             color = 'red')

plt.xlabel('Cigarettes per day')



########################

plt.tight_layout()
plt.savefig('omaps, fmaps, cigs.png')

plt.show()

################################################
#Combining outliers with thresholds gathered from qualitative analysis
################################################

bwt_df_quantiles = bwt_df.loc[:, :].quantile([0.10,
                                                0.40,
                                                0.60,
                                                0.80,
                                                0.95])

print(bwt_df_quantiles)

# apgar score
omaps_lo = 3
fmaps_lo = 3

# average cigarettes per day
cigs_hi = 1

# average cigarettes per day
cigs_hi = 1

#APGAR Scores
bwt_df['out_omaps'] = 0
for val in enumerate(bwt_df.loc[ : , 'omaps']):
    if val[1] <= omaps_lo:
        bwt_df.loc[val[0], 'out_omaps'] = 1

bwt_df['out_fmaps'] = 0
for val in enumerate(bwt_df.loc[ : , 'fmaps']):
    if val[1] <= fmaps_lo:
        bwt_df.loc[val[0], 'out_fmaps'] = 1

#Cigarettes per day
bwt_df['out_cigs'] = 0
for val in enumerate(bwt_df.loc[ : , 'cigs']):
    if val[1] >= cigs_hi:
        bwt_df.loc[val[0], 'out_cigs'] = 1
        

####################################
# Correlation

df_corr = bwt_df.corr().round(2)

print(df_corr)

df_corr.loc['bwght'].sort_values(ascending = False)

# Correlation Matrix
# Using palplot

sns.palplot(sns.color_palette('coolwarm', 12))

fig, ax = plt.subplots(figsize=(15,15))

df_corr2 = df_corr.iloc[1:19, 1:19]

sns.heatmap(df_corr2,
            cmap = 'coolwarm',
            square = True,
            annot = True,
            linecolor = 'black',
            linewidths = 0.5)
plt.show()

#####################################
# Saving file
bwt_df.to_excel('birthweight_new.xlsx')

