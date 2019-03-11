# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 16:52:40 2019

@author: lucas
"""

#importing the necessary packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#displaying all the columns when called
pd.set_option('display.max_columns', 500)

df = pd.read_excel('birthweight_feature_set.xlsx')


#Dataset exploration

df.columns


df.shape


df.describe().round(2)


df.info()


df.corr()

################################################################
#Flagging missing values 
################################################################

df.isnull().sum()

for col in df:

    """ Create columns that are 0s if a value was not missing and 1 if
    a value is missing. """
    
    if df[col].isnull().any():
        df['m_'+col] = df[col].isnull().astype(int)
        
df_dropped = df.dropna()

################################################################
#checking missing values from number of prenatal visits
################################################################

fig, ax = plt.subplots(figsize=(20,10))
sns.distplot(df_dropped['npvis'])

df['npvis'].describe()


### npvis is relatively normal.
### Only 3 missing values. Taking in consideration that the factors that will 
#probably influence the msot on the birth weight are cigaretts and drinks, and
#taking in consideration that these have extremely low correlation with number of visits(plus
#the low correlation with the birthweight), I'll be inputing the median 
#to speed-up the analysis. Inputing the median will most likely have little
#to no influence on the final model. 

fill = df['npvis'].median()

df['npvis'] = df['npvis'].fillna(fill)



################################################################
#checking missing values from father's education
################################################################


df['feduc'].describe()

fig, ax = plt.subplots(figsize=(20,10))
sns.distplot(df_dropped['feduc'])

### The larger amount of variables is concentrated around the mean and the max value (17)
# To continue with the trend, I'll input the mean for the missing values.


fill = df['feduc'].mean()

df['feduc'] = df['feduc'].fillna(fill)




################################################################
#General Outlier Analysis
################################################################


"""
Prenatal visits, father's education & father's age are all continuous variables
"""

############


plt.subplot(2, 2, 1)
sns.distplot(df['npvis'],
             bins = 'fd',
             color = 'g')

plt.xlabel('prenatal visits')


########################


plt.subplot(2, 2, 2)
sns.distplot(df['feduc'],
             bins = 'fd',
             color = 'y')

plt.xlabel('''Father's education''')



########################


plt.subplot(2, 2, 3)
sns.distplot(df['fage'],
             bins = 'fd',
             color = 'orange')

plt.xlabel('''Father's age''')



########################

plt.tight_layout()
plt.savefig('npvis, fage, feduc.png')

plt.show()


'''
Pre-conclusions:

Prenatal visits: <5
                 >20
                 

Father's Education: <7
                    > x


Father's age: < x
              > 65

'''
################################################
#Combining outliers with thresholds gathered from qualitative analysis
################################################


df_2_quantiles = df.loc[:, :].quantile([0.10,
                                                0.40,
                                                0.60,
                                                0.80,
                                                0.95])

print(df_2_quantiles)




'''

Final Threshold/outlier:
    
Prenatal Visit low: 9
Prenatal Visit high: 15

According to qualitative analysis/research, this threshold is the healthy 
interval in which a mother should go to prenatal care. Normally the minimum 
threshold would be 10 visits, but let's assume 9 visits due to 1 month delay
in discovering the pregnancy. Also, more than 15 is signicant of probable
pregnancy complications.




Father's Education low: 11 (10%)
There wasn't research evidence that this could influence greatly on the variable
so this threshold was decided by outlier analysis.





Father's Age: > 45
According to the research, higher paternal age is associated with higher incidence
of premature birth, low birth weight, and others. According to the research, 45
years should be the threshold.

'''


###Flagging outliers/threshold

npvis_lo = 9

npvis_hi = 15

feduc_lo = 11

fage_hi = 45


########################
# Prenatal visits

df['out_npvis'] = 0


for val in enumerate(df.loc[ : , 'npvis']):
    
    if val[1] > npvis_hi:
        df.loc[val[0], 'out_npvis'] = 1
        
    

for val in enumerate(df.loc[ : , 'npvis']):
    
    if val[1] < npvis_lo:
        df.loc[val[0], 'out_npvis'] = -1
        


########################
# Father's Education
        
df['out_feduc'] = 0

for val in enumerate(df.loc[ : , 'feduc']):
    
    if val[1] <= feduc_lo:
        df.loc[val[0], 'out_feduc'] = -1
            

  
########################
# Father's age
        
df['out_fage'] = 0

for val in enumerate(df.loc[ : , 'fage']):
    
    if val[1] >= fage_hi:
        df.loc[val[0], 'out_fage'] = 1
    
    
    
    
########################################################
#Dataset still needs the treatment of the other missing values to begin modeling.
########################################################
       

    








