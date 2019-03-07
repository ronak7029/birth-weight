#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 17:33:50 2018

@author: chase.kusterer

Working Directory:
/Users/chase.kusterer/Desktop/Chase/Special

Purpose:
    To review concepts covered in Python for Data Science.
"""


# Loading Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


file = 'Ames Housing Dataset.xls'


housing = pd.read_excel(file)



########################
# Fundamental Dataset Exploration
########################

# Column names
housing.columns


# Displaying the first rows of the DataFrame
print(housing.head())


# Dimensions of the DataFrame
housing.shape


# Information about each variable
housing.info()


# Descriptive statistics
housing.describe().round(2)


housing.sort_values('SalePrice', ascending = False)



###############################################################################
# Imputing Missing Values
###############################################################################

print(
      housing
      .isnull()
      .sum()
      )



for col in housing:

    """ Create columns that are 0s if a value was not missing and 1 if
    a value is missing. """
    
    if housing[col].isnull().any():
        housing['m_'+col] = housing[col].isnull().astype(int)



# Not a lot of missing values in most columns. Mas Vnr Area should be
# explored further


# Creating a dropped dataset to graph 'Mas Vnr Area'
df_dropped = housing.dropna()

sns.distplot(df_dropped['Mas Vnr Area'])


# 'Mas Vnr Area' is zero inflated. Imputing with zero.
fill = 0

housing['Mas Vnr Area'] = housing['Mas Vnr Area'].fillna(fill)




# Everything else being filled with the median
fill = housing['Total Bsmt SF'].median()

housing['Total Bsmt SF'] = housing['Total Bsmt SF'].fillna(fill)



fill = housing['Garage Cars'].median()

housing['Garage Cars'] = housing['Garage Cars'].fillna(fill)



fill = housing['Garage Area'].median()

housing['Garage Area'] = housing['Garage Area'].fillna(fill)



# Checking the overall dataset to see if there are any remaining
# missing values
print(
      housing
      .isnull()
      .any()
      .any()
      )

#any.() is a aggregation function


###############################################################################
# Outlier Analysis
###############################################################################

housing_quantiles = housing.loc[:, :].quantile([0.20,
                                                0.40,
                                                0.60,
                                                0.80,
                                                1.00])

    
print(housing_quantiles)



for col in housing:
    print(col)



"""

Assumed Continuous/Interval Variables - 

Lot Area
Overall Qual
Overall Cond
Mas Vnr Area
Total Bsmt SF
1st Flr SF
2nd Flr SF
Gr Liv Area
Full Bath
Half Bath
Bedroom Abv Gr
Kitchen AbvGr
TotRms AbvGrd
Fireplaces
Grarage Cars
Garage Area
Pool Area
SalePrice



Assumed Categorical -

Street
Lot Config
Neighborhood
House Style
Year Built
Roof Style



Binary Classifiers -

m_Mas Vnr Area
m_Total Bsmt SF
m_Garage Cars
m_Garage Area

"""



########################
# Visual EDA (Histograms)
########################


plt.subplot(2, 2, 1)
sns.distplot(housing['Lot Area'],
             bins = 35,
             color = 'g')

plt.xlabel('Lot Area')


########################


plt.subplot(2, 2, 2)
sns.distplot(housing['Overall Qual'],
             bins = 30,
             color = 'y')

plt.xlabel('Overall Qual')



########################


plt.subplot(2, 2, 3)
sns.distplot(housing['Overall Cond'],
             bins = 17,
             kde = False,
             rug = True,
             color = 'orange')

plt.xlabel('Overall Cond')



########################


plt.subplot(2, 2, 4)

sns.distplot(housing['Mas Vnr Area'],
             bins = 17,
             kde = False,
             rug = True,
             color = 'r')

plt.xlabel('Mas Vnr Area')



plt.tight_layout()
plt.savefig('Housing Data Histograms 1 of 5.png')

plt.show()



########################
########################



plt.subplot(2, 2, 1)
sns.distplot(housing['1st Flr SF'],
             bins = 35,
             color = 'g')

plt.xlabel('1st Flr SF')


########################


plt.subplot(2, 2, 2)
sns.distplot(housing['2nd Flr SF'],
             bins = 30,
             color = 'y')

plt.xlabel('2nd Flr SF')



########################


plt.subplot(2, 2, 3)
sns.distplot(housing['Full Bath'],
             bins = 17,
             kde = False,
             rug = True,
             color = 'orange')

plt.xlabel('Full Bath')



########################


plt.subplot(2, 2, 4)

sns.distplot(housing['Half Bath'],
             bins = 17,
             kde = False,
             rug = True,
             color = 'r')

plt.xlabel('Half Bath')



plt.tight_layout()
plt.savefig('Housing Data Histograms 2 of 5.png')

plt.show()



########################
########################



plt.subplot(2, 2, 1)
sns.distplot(housing['Kitchen AbvGr'],
             bins = 30,
             color = 'y')

plt.xlabel('Kitchen AbvGr')



########################

plt.subplot(2, 2, 2)
sns.distplot(housing['TotRms AbvGrd'],
             bins = 17,
             kde = False,
             rug = True,
             color = 'orange')

plt.xlabel('TotRms AbvGrd')



########################

plt.subplot(2, 2, 3)

sns.distplot(housing['Fireplaces'],
             bins = 17,
             kde = False,
             rug = True,
             color = 'r')

plt.xlabel('Fireplaces')



########################

plt.subplot(2, 2, 4)
sns.distplot(housing['Garage Cars'],
             bins = 35,
             color = 'g')

plt.xlabel('Garage Cars')



plt.tight_layout()
plt.savefig('Housing Data Histograms 3 of 5.png')

plt.show()



########################
########################



plt.subplot(2, 2, 1)
sns.distplot(housing['Garage Area'],
             bins = 30,
             color = 'y')

plt.xlabel('Garage Area')



########################


plt.subplot(2, 2, 2)
sns.distplot(housing['Pool Area'],
             bins = 17,
             kde = False,
             rug = True,
             color = 'orange')

plt.xlabel('Pool Area')



########################

plt.subplot(2, 2, 3)

sns.distplot(housing['SalePrice'],
             bins = 17,
             kde = False,
             rug = True,
             color = 'r')

plt.xlabel('SalePrice')



plt.subplot(2, 2, 4)
sns.distplot(housing['Total Bsmt SF'],
             bins = 35,
             color = 'g')

plt.xlabel('Total Bsmt SF')



plt.tight_layout()
plt.savefig('Housing Data Histograms 4 of 5.png')

plt.show()



########################
########################



plt.subplot(2, 2, 1)

sns.distplot(housing['Gr Liv Area'],
             bins = 17,
             kde = False,
             rug = True,
             color = 'r')

plt.xlabel('Gr Liv Area')



plt.subplot(2, 2, 2)

sns.distplot(housing['Porch Area'],
             bins = 17,
             kde = False,
             rug = True,
             color = 'r')

plt.xlabel('Porch Area')



plt.tight_layout()
plt.savefig('Housing Data Histograms 5 of 5.png')

plt.show()



########################
# Tuning and Flagging Outliers
########################

housing_quantiles = housing.loc[:, :].quantile([0.05,
                                                0.40,
                                                0.60,
                                                0.80,
                                                0.95])


# Outlier flags
lot_area_hi      = 15000

overall_qual_lo  = 2

overall_qual_hi  = 10

overall_cond_lo  = 3

mas_vnr_area_hi  = 150

Total_Bsmt_SF_lo = 400

Total_Bsmt_SF_hi = 2000

fst_Flr_SF_hi    = 1700

sec_Flr_SF_lo    = 0

sec_Flr_SF_hi    = 1100

Full_Bath_hi     = 3

Half_Bath_hi     = 2

Kitchen_AbvGr_hi = 2

TotRms_AbvGr_hi  = 9

Fireplaces_hi    = 2

Garage_Cars_hi   = 4

Garage_Area_lo   = 0

Garage_Area_hi   = 856

Porch_Area_lo    = 0

Porch_Area_hi    = 469.1

Pool_Area_lo     = 0

Gr_Liv_Area_lo   = 861

Gr_Liv_Area_hi   = 2500

SalePrice_hi     = 300000



########################
# Creating Outlier Flags
########################

# Building loops for outlier imputation



########################
# Lot Area

housing['out_Lot_Area'] = 0


for val in enumerate(housing.loc[ : , 'Lot Area']):
    
    if val[1] >= lot_area_hi:
        housing.loc[val[0], 'out_Lot_Area'] = 1

# See Footnote 0 for a more detailed explaination of the above code.


########################
# Overall Qual

housing['out_Overall_Qual'] = 0


for val in enumerate(housing.loc[ : , 'Overall Qual']):
    
    if val[1] <= overall_qual_lo:
        housing.loc[val[0], 'out_Overall_Qual'] = -1



for val in enumerate(housing.loc[ : , 'Overall Qual']):
    
    if val[1] >= overall_qual_hi:
        housing.loc[val[0], 'out_Overall_Qual'] = 1
        
        
        
########################
# Overall Cond

housing['out_Overall_Cond'] = 0


for val in enumerate(housing.loc[ : , 'Overall Cond']):
    
    if val[1] <= overall_cond_lo:
        housing.loc[val[0], 'out_Overall_Cond'] = -1



########################
# Mas Vnr Area
     
housing['out_Mas_Vnr_Area'] = 0


for val in enumerate(housing.loc[ : , 'Mas Vnr Area']):
    
    if val[1] >= mas_vnr_area_hi:
        housing.loc[val[0], 'out_Mas_Vnr_Area'] = 1



########################
# Total Bsmt SF

housing['out_Total_Bsmt_SF'] = 0


for val in enumerate(housing.loc[ : , 'Total Bsmt SF']):
    
    if val[1] <= Total_Bsmt_SF_lo:
        housing.loc[val[0], 'out_Total_Bsmt_SF'] = -1


for val in enumerate(housing.loc[ : , 'Total Bsmt SF']):
    
    if val[1] >= Total_Bsmt_SF_hi:
        housing.loc[val[0], 'out_Total_Bsmt_SF'] = 1



########################
# 1st Flr SF

housing['out_ff_SF'] = 0


for val in enumerate(housing.loc[ : , '1st Flr SF']):
    
    if val[1] >= fst_Flr_SF_hi:
        housing.loc[val[0], 'out_ff_SF'] = 1



########################
# 2nd Flr SF

housing['out_sf_SF'] = 0


for val in enumerate(housing.loc[ : , '2nd Flr SF']):
    
    if val[1] <= sec_Flr_SF_lo:
        housing.loc[val[0], 'out_sf_SF'] = -1



for val in enumerate(housing.loc[ : , '2nd Flr SF']):
    
    if val[1] >= sec_Flr_SF_hi:
        housing.loc[val[0], 'out_sf_SF'] = 1



########################
# Full Bath

housing['out_Full_Bath'] = 0

for val in enumerate(housing.loc[ : , 'Full Bath']):
    
    if val[1] >= Full_Bath_hi:
        housing.loc[val[0], 'out_Full_Bath'] = 1



########################
# Half Bath

housing['out_Half_Bath'] = 0

for val in enumerate(housing.loc[ : , 'Half Bath']):
    
    if val[1] >= Half_Bath_hi:
        housing.loc[val[0], 'out_Half_Bath'] = 1



########################
# Kitchen AbvGr

housing['out_Kitchen_AbvGr'] = 0

for val in enumerate(housing.loc[ : , 'Kitchen AbvGr']):
    
    if val[1] >= Kitchen_AbvGr_hi:
        housing.loc[val[0], 'out_Kitchen_AbvGr'] = 1



########################
# TotRms AbvGrd

housing['out_TotRms_AbvGrd'] = 0

for val in enumerate(housing.loc[ : , 'TotRms AbvGrd']):
    
    if val[1] >= TotRms_AbvGr_hi:
        housing.loc[val[0], 'out_TotRms_AbvGrd'] = 1



########################
# Fireplaces

housing['out_Fireplaces'] = 0

for val in enumerate(housing.loc[ : , 'Fireplaces']):
    
    if val[1] >= Fireplaces_hi:
        housing.loc[val[0], 'out_Fireplaces'] = 1



########################
# Garage Cars

housing['out_Garage_Cars'] = 0

for val in enumerate(housing.loc[ : , 'Garage Cars']):
    
    if val[1] >= Garage_Cars_hi:
        housing.loc[val[0], 'out_Garage_Cars'] = 1



########################
# Garage Area

housing['out_Garage_Area'] = 0


for val in enumerate(housing.loc[ : , 'Garage Area']):
    
    if val[1] <= Garage_Area_lo:
        housing.loc[val[0], 'out_Garage_Area'] = -1



for val in enumerate(housing.loc[ : , 'Garage Area']):
    
    if val[1] >= Garage_Area_hi:
        housing.loc[val[0], 'out_Garage_Area'] = 1



########################
# Porch Area

housing['out_Porch_Area'] = 0


for val in enumerate(housing.loc[ : , 'Porch Area']):
    
    if val[1] <= Porch_Area_lo:
        housing.loc[val[0], 'out_Porch_Area'] = -1



for val in enumerate(housing.loc[ : , 'Porch Area']):
    
    if val[1] >= Porch_Area_hi:
        housing.loc[val[0], 'out_Porch_Area'] = 1



########################
# Pool Area

housing['out_Pool_Area'] = 0


for val in enumerate(housing.loc[ : , 'Pool Area']):
    
    if val[1] <= Pool_Area_lo:
        housing.loc[val[0], 'out_Pool_Area'] = 1



########################
# Gr Liv Area

housing['out_Gr_Liv_Area'] = 0


for val in enumerate(housing.loc[ : , 'Gr Liv Area']):
    
    if val[1] <= Gr_Liv_Area_lo:
        housing.loc[val[0], 'out_Gr_Liv_Area'] = -1



for val in enumerate(housing.loc[ : , 'Gr Liv Area']):
    
    if val[1] >= Gr_Liv_Area_hi:
        housing.loc[val[0], 'out_Gr_Liv_Area'] = 1



########################
# SalePrice

housing['out_Sale_Price'] = 0


for val in enumerate(housing.loc[ : , 'SalePrice']):
    
    if val[1] >= SalePrice_hi:
        housing.loc[val[0], 'out_Sale_Price'] = 1




###############################################################################
# Qualitative Variable Analysis (Boxplots)
###############################################################################
        
"""

Assumed Categorical -

Street
Lot Config
Neighborhood

"""

########################
# Street


housing.boxplot(column = ['SalePrice'],
                by = ['Street'],
                vert = False,
                patch_artist = False,
                meanline = True,
                showmeans = True)


plt.title("Sales Price by Street Type")
plt.suptitle("")

plt.savefig("Sales Price by Street Type.png")

plt.show()



########################
# Lot Config


housing.boxplot(column = ['SalePrice'],
                by = ['Lot Config'],
                vert = False,
                 patch_artist = False,
                 meanline = True,
                 showmeans = True)


plt.title("Sales Price by Lot Configuration")
plt.suptitle("")

plt.savefig("Sales Price by Lot Configuration.png")

plt.show()



########################
# Neighborhood


housing.boxplot(column = ['SalePrice'],
                by = ['Neighborhood'],
                vert = False,
                 patch_artist = False,
                 meanline = True,
                 showmeans = True)


plt.title("Sales Price by Neighborhood")
plt.suptitle("")

plt.savefig("Sales Price by Neighborhood.png")

plt.show()



###############################################################################
# Correlation Analysis
###############################################################################

housing.head()


df_corr = housing.corr().round(2)


print(df_corr)


df_corr.loc['SalePrice'].sort_values(ascending = False)



########################
# Correlation Heatmap
########################

# Using palplot to view a color scheme
sns.palplot(sns.color_palette('coolwarm', 12))

fig, ax = plt.subplots(figsize=(15,15))

df_corr2 = df_corr.iloc[1:19, 1:19]

sns.heatmap(df_corr2,
            cmap = 'coolwarm',
            square = True,
            annot = True,
            linecolor = 'black',
            linewidths = 0.5)


plt.savefig('Housing Correlation Heatmap.png')
plt.show()

housing.to_excel('Ames_explored.xlsx')



"""
###############################################################################
# Footnotes
###############################################################################
        
Footnote 0: for loop with enumerate

housing['out_Lot_Area'] = 0                  # creating a new column called out_Lot_Area


for val in enumerate(                        # starting a for loop where indexes are also called
housing.loc[ : , 'Lot Area']):               # on all rows of the column price in the housing dataset
    
if val[1] > lot_area_hi:                     # if val[1] (i.e. the value of Lot Area) meets this condition
diamonds.loc[val[0], 'out_Lot_Area'] = 1     # out_Lot_Area = 1 at the same index (i.e. val[0])


*******************************************************************************
"""