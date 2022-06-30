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
# print("---------------------------------------------------")

m = 2 # only plot if there are more than m values per category Z

# require a material to have more than m data points to plot it
count_z = df[Z].value_counts()
filt_minpoints = df[Z].isin(count_z[count_z > m].index)

# require more than m unique temperatures to plot the material
count_x = df.groupby(Z)[X].nunique()
filt_temps = df[Z].isin(count_x[count_x > m].index)

filt = filt_minpoints & filt_temps

df_plot = df.loc[filt, :] # create new df with filtered values
# print(df_plot.tail())

# print(count_x)
# print(filt_minpoints)
# print(filt_temps)
print("---------------------------------------------------")



# print(df[Z].value_counts(), "\n")
print(df_plot[Z].value_counts(), "\n")
print(df_plot.groupby(Z)[X].nunique(), "\n")


# PLOT

# df.plot(x=X, y=Y, marker='x', kind='scatter') # scatter plot
# sns.scatterplot(data=df, x=X, y=Y, hue=Z) # scatter plot coloured by material
# sns.regplot(data=df, x=X, y=Y, ci=None) # scatter plot with linear regression
sns.lmplot(data=df_plot, x=X, y=Y, hue=Z, ci=None, scatter_kws={"s": 10, "alpha":0.4}, line_kws={"lw":1.25, "alpha":0.5}) # scatter plot with linear regression for each category Z 

# SET PLOT TITLE
plt.title("Density vs Temperature ALD Thin Films")

# SET AXIS LABELS
plt.xlabel("Deposition Temperature (°C)")
plt.ylabel("Density $(g.cm^{-3})$")


# plt.show()

# bbox_inches stops title from being cut off in the plot
plt.savefig('plots/plot1.png', dpi=150, bbox_inches="tight")
