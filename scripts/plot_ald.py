# Daria Zachariassen
# June 23, 2022

"""
ALD Data Scraping Project

Reads data from google sheet.
"""

import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt


sheet_id = "1CnYIYPMymwAKaVlElBk4ceNN2RtTxpKIHTzuTRjfY3s"
sheet_name = "FilmProps"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"

# LOAD DATA
df = pd.read_csv(url)

# CLEAN DATA
df.dropna(axis='columns', how='all', inplace=True) # drop columns that are all NA values

df.columns = df.columns.str.replace(r'[ ]{2,}',' ', regex=True) # make sure column names only have 1 space between words


# VIEW DATA
print(list(df.columns), "\n") # view column names

print(df.info(), "\n") # view column names and types
print(df.head(), "\n") # view the first few rows of the dataframe

# VARIABLES
X="Tdep (°C)"
Y="Density (g.cm-3)"
Z="Material"


# FILTER

m = 2 # only plot if there are more than m values per category Z
count_z = df[Z].value_counts()
filt_minpoints = df[Z].isin(count_z[count_z > m].index)

df_plot = df.loc[filt_minpoints, :] # create new df with filtered values
# print(df_plot.tail())

print(df[Z].value_counts(), "\n")
print(df_plot[Z].value_counts(), "\n")


# PLOT

# df.plot(x=X, y=Y, marker='x', kind='scatter') # scatter plot
# sns.scatterplot(data=df, x=X, y=Y, hue=Z) # scatter plot coloured by material
# sns.regplot(data=df, x=X, y=Y, ci=None) # scatter plot with linear regression
sns.lmplot(data=df_plot, x=X, y=Y, hue=Z, ci=None) # scatter plot with linear regression for each category Z 

# plt.show()
plt.savefig('plots/plot1.png', dpi=150)
