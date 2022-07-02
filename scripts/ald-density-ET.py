"""
Plot Experimental vs Theoretical Densities

https://docs.google.com/spreadsheets/d/1CnYIYPMymwAKaVlElBk4ceNN2RtTxpKIHTzuTRjfY3s/edit#gid=1436201923
"""


import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt

sheet_name_exp = "FilmProps"
sheet_name_thr = "TheorDensity"
sheet_id = "1CnYIYPMymwAKaVlElBk4ceNN2RtTxpKIHTzuTRjfY3s"
url_exp = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name_exp}"
url_thr = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name_thr}"

# url_exp = "data/data_exp.csv"
# url_thr = "data/data_thr.csv"

# LOAD DATA
df_exp = pd.read_csv(url_exp)
df_thr = pd.read_csv(url_thr)

# df_exp.to_csv("data/data_exp.csv", encoding='utf-8')
# df_thr.to_csv("data/data_thr.csv", encoding='utf-8')

# CLEAN DATA
df_exp.dropna(axis='columns', how='all', inplace=True) # drop columns that are all NA values
df_exp.columns = df_exp.columns.str.replace(r'[ ]{2,}',' ', regex=True) # make sure column names only have 1 space between words

df_thr.dropna(axis='columns', how='all', inplace=True) # drop columns that are all NA values
df_thr.columns = df_thr.columns.str.replace(r'[ ]{2,}',' ', regex=True) # make sure column names only have 1 space between words


# VIEW DATA
print(list(df_exp.columns), "\n") # view column names
print(df_exp.info(), "\n") # view column names and types
# print(df_exp.head(), "\n") # view the first few rows of the dataframe

print("-"*30)

print(list(df_thr.columns), "\n") # view column names
print(df_thr.info(), "\n") # view column names and types
# print(df_thr.head(), "\n") # view the first few rows of the dataframe



# GET DATA TO PLOT

exp_density = df_exp.loc[:,["Material", "Density (g.cm-3)"]]
thr_density = df_thr.loc[:,["Material", "Density gcm3"]]

exp_density.rename(columns={"Density (g.cm-3)": "Experimental Density"}, inplace=True)
thr_density.rename(columns={"Density gcm3": "Theoretical Density"}, inplace=True)

print(exp_density.head())
print(thr_density.head())

df_merged = pd.merge(exp_density, thr_density, on="Material")

print(df_merged.info())




# PLOT

def plot_data(df, x, y, z):

    fig, ax = plt.subplots(figsize=(10,8)) # figsize=(6, 7)
    # ax.grid(True, color = '#e8e8e6', linestyle = '--', linewidth = 0.5)
    # ax = fig.add_subplot()
    # ax = sns.lineplot(data=df, x=x, y=y, hue=z, marker="o", ci=None, markersize=4, alpha = 0.9, linestyle='')
    # sns.lineplot(data=df, x=x, y=y, ax=ax, sort=False, hue=z, dashes=False, alpha=0.3, legend=False, ci=None)
    order_z = list(df.sort_values(by=[z], ascending=True)[z].unique())
    # random.Random(2).shuffle(order_z)
    # print(order_z)

    # g2 = sns.lineplot(ax=ax, data=df, x=x, y=y, hue=z, alpha=0.4, lw=1, ci=None, legend=False, hue_order=order_z)
    ax.axline((0,0), slope=1, color='silver', lw=1.5, label='_none_')
    g1 = sns.scatterplot(ax=ax, data=df, x=x, y=y, hue=z, alpha=0.6, s=35, style=z, hue_order=order_z, style_order=order_z)

    return fig


# df_merged.plot(x="Theoretical Density", y="Experimental Density", marker='x', kind='scatter') # scatter plot
fig = plot_data(df_merged, x="Theoretical Density", y="Experimental Density", z="Material")


plt.savefig('plots/plotDensities.png', dpi=200, bbox_inches="tight")

# plt.show()

