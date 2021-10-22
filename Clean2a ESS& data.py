import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 2a: label missing data as nans and chuck them

data_raw = pd.read_csv('ESS7_cleaned1.csv')
data= data_raw

data.head()

data.isna().sum().sort_values(ascending = False)*100 # calculate % of data missing for each column

#data.hist(grid=False, bins = 10)
#plt.show()

data.loc[data.height > 776,'height'] = np.nan
data.loc[data.weight > 776,'weight'] = np.nan
data.loc[data.age > 998,'age'] = np.nan
data.loc[data.years_education > 76,'years_education'] = np.nan
data.loc[data.industry > 665,'industry'] = np.nan

#data['political_orientation_binned']  = pd.cut(x = data['political_orientation_lr'], bins=[0,4,6, 10]) # 11% of people did give political orientation, perhaps bin it?
#sns.countplot(y = 'political_orientation_binned', data = data)

data.isna().sum().sort_values(ascending = False)*100 # calculate % of data missing for each column
data.isna().mean().round(3).sort_values(ascending = False)*100 # calculate % of data missing for each column

data = data.drop(columns = ['binge_drinking','childhood_financial_problems','family_conflict_childhood','hours_overtime_excl','hours_overtime_incl'])

data.isna().sum().sort_values(ascending = False)*100 # calculate % of data missing for each column
data.isna().mean().round(3).sort_values(ascending = False)*100 # calculate % of data missing for each column

#data.loc[:,'income_decile_impute'] = data.income_decile
#data = data.fillna({'income_decile_impute':5})
#data.loc[:,'income_decile_minus'] = data.income_decile
#data = data.fillna({'income_decile_minus':-1})
#data = data.drop(columns = ['income_decile'])

data = data.dropna()

data.shape
data.columns

data.to_csv('ESS7_cleaned2a.csv',index = False)
