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
# # print(df_exp.info(), "\n") # view column names and types
# # print(df_exp.head(), "\n") # view the first few rows of the dataframe
# print(df_exp["Material"].value_counts())
# print("-"*60) # visual separator on terminal

print(list(df_thr.columns), "\n") # view column names
# # print(df_thr.info(), "\n") # view column names and types
# # print(df_thr.head(), "\n") # view the first few rows of the dataframe
# print(df_thr["Material"].value_counts())
print("-"*60) # visual separator on terminal


# GET DATA TO PLOT

exp_density = df_exp.loc[:,["Material", "Density (g.cm-3)"]]
thr_density = df_thr.loc[:,["Material", "Density gcm3"]]

NAME_EXP = "experimental"
NAME_THR = "theoretical"

exp_density.rename(columns={"Density (g.cm-3)": NAME_EXP}, inplace=True)
thr_density.rename(columns={"Density gcm3": NAME_THR}, inplace=True)

# print(exp_density.head())
# print(thr_density.head())
# print("-"*60) # visual separator on terminal

df_merged = pd.merge(exp_density, thr_density, on="Material")

# print(df_merged["Material"].value_counts())
# print(df_merged.info())
# print(df_merged.head())
# print("-"*60) # visual separator on terminal

def get_tidy_df(df1, df2, cat, val, idv):

    df_m1 = pd.melt(df1, id_vars=idv, var_name=cat, value_name=val)
    df_m1.dropna(inplace=True)
    # print(df_m1.info())

    df_m2 = pd.melt(df2, id_vars=idv, var_name=cat, value_name=val)
    df_m2.dropna(inplace=True)
    # print(df_m2.info())

    df_merged = pd.concat([df_m1, df_m2], ignore_index=True)
    # print(df_merged.info())

    return df_merged

df_tidy = get_tidy_df(exp_density, thr_density, cat="Type", val="Density", idv="Material")
print(df_tidy.info())
print(df_tidy.head())

def tidy_handler(df, group, y):
    gb = df.groupby(group)
    df_mean = gb[y].agg(np.mean)
    df_mean = df_mean.reset_index()
    print(df_mean.head(20), "\n")
    print(df_mean.info(), "\n")

# tidy_handler(df_tidy, ["Type", "Material"], y="Density")

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
    ax.axline((0,0), slope=1, color='silver', lw=1, ls='--', label='_none_')
    g1 = sns.scatterplot(ax=ax, data=df, x=x, y=y, hue=z, alpha=0.6, s=35, style=z, hue_order=order_z, style_order=order_z)

    # for line in range(0,df.shape[0]):
    #      plt.text(df[x][line], 0, df[z][line], horizontalalignment='center', size='small', color='black', weight='light')


    return fig


def plot_swarm(df, x, y, z):
    # fig, ax = plt.subplots(figsize=(10,8)) # figsize=(6, 7)

    order_z = list(df.sort_values(by=[z], ascending=True)[z].unique())


    g1 = sns.catplot(data=df, x=y, y=y, hue=x, col=z, col_wrap=7, alpha=0.8, sharex=False, sharey=False, kind="strip", jitter=False, height=5, aspect=1.25, ci=None)
    # plt.xticks(rotation=90)

    # ax1 = g1.axes[0]

    # g1.set(ylim=(0,None), xlim=(0,None))

    for ax in g1.axes.flatten():
        ax.axline((0,0), slope=1, color='silver', lw=1, ls='--', label='_none_')
        ax.tick_params(labelbottom=True, labelleft=True)

        ax.set(xlim=(0,12))  # x axis starts at 0
        ax.set(ylim=(0,12))  # y axis starts at 0




    return g1



# # df_merged.plot(x=NAME_THR, y=NAME_EXP, marker='x', kind='scatter') # scatter plot
# fig1 = plot_data(df_merged, x=NAME_THR, y=NAME_EXP, z="Material")
fig2 = plot_swarm(df_tidy, x="Type", y="Density", z="Material")


# fig1.savefig('plots/plotDensities.png', dpi=200, bbox_inches="tight")
fig2.savefig('plots/plotDensities-2.png', dpi=300, bbox_inches="tight")

# plt.show()

