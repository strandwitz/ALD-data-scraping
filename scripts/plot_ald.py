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
from matplotlib.ticker import AutoMinorLocator, FuncFormatter
import random




sheet_id = "1CnYIYPMymwAKaVlElBk4ceNN2RtTxpKIHTzuTRjfY3s"
sheet_name = "FilmProps"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"

# TODO: option to choose material. Al2O3, TiO2


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
df_plot2 = df.loc[filt_minpoints, :] # create new df with filtered values

# print(df_plot.tail())

# print(count_x)
# print(filt_minpoints)
# print(filt_temps)
print("---------------------------------------------------")



# print(df[Z].value_counts(), "\n")
print(df_plot2[Z].value_counts(), "\n")
print(df_plot2.groupby(Z)[X].nunique(), "\n")


def get_line_plot(df, x, y, z):
    # sns.set(style='darkgrid')
    # sns.set(rc={'axes.facecolor':'aliceblue', 'axes.edgecolor':'grey'})
    # sns.set(rc={"xtick.bottom" : True, "ytick.left" : True, 
    #     "xtick.color" : 'silver', "ytick.color" : 'silver'})



    fig, ax = plt.subplots() # figsize=(6, 7)
    ax.grid(True, color = '#e8e8e6', linestyle = '--', linewidth = 0.5)
    # ax = fig.add_subplot()
    # ax = sns.lineplot(data=df, x=x, y=y, hue=z, marker="o", ci=None, markersize=4, alpha = 0.9, linestyle='')
    # sns.lineplot(data=df, x=x, y=y, ax=ax, sort=False, hue=z, dashes=False, alpha=0.3, legend=False, ci=None)
    order_z = list(df.sort_values(by=[Z], ascending=True)[Z].unique())
    # random.Random(2).shuffle(order_z)
    print(order_z)

    g2 = sns.lineplot(ax=ax, data=df, x=x, y=y, hue=z, alpha=0.4, lw=1, ci=None, legend=False, hue_order=order_z)
    g1 = sns.scatterplot(ax=ax, data=df, x=x, y=y, hue=z, alpha=0.7, s=25, style=z, hue_order=order_z, style_order=order_z)

    # SET PLOT TITLE
    plt.title("Density vs Temperature ALD Thin Films")

    # SET AXIS LABELS
    plt.xlabel("Deposition Temperature (°C)")
    plt.ylabel("Density $(g.cm^{-3})$")


    # df.groupby([x,z]).count()[y].unstack().plot(ax=ax, marker="o", markersize=4, alpha=0.6)

    # plt.scatter(df[x], df[y])

    # for label, df_g in df.groupby(z):
    #     df_g.vals.plot(kind="kde", ax=ax, label=label)

    plt.legend(title=f"{z}s", loc=2, bbox_to_anchor=(1.01, 1));


    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))

    return fig

def get_fit_plot(df, x, y, z):
    # plot
    print(df.info())

    # PLOT
    # fig1, ax1 = plt.subplots()
    # fig2, ax2 = plt.subplots()
    # fig = plt.figure()
    # ax = fig.add_subplot()

    # df.plot(x=X, y=Y, marker='x', kind='scatter') # scatter plot
    # sns.scatterplot(data=df, x=X, y=Y, hue=Z) # scatter plot coloured by material
    # sns.regplot(data=df, x=X, y=Y, ci=None) # scatter plot with linear regression
    
    
    sns.set(font_scale = 1.50)

    # scatter plot with linear regression for each category Z 
    fig = sns.lmplot(data=df, x=x, y=y, hue="PEALD?", markers=["^", "v"], col=z, col_wrap=4, ci=None, 
        facet_kws = dict(sharex=False, sharey=False), 
        scatter_kws={"s": 45, "alpha":0.7}, 
        line_kws={"lw":1.5, "alpha":0.5})

    fig.set_titles(col_template="{col_name}")

    # TODO: fix axis sigfigs

    for ax in fig.axes.flat:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, p: f'{y:.2f}'))
    
    # splot2 = sns.lmplot(data=df_plot, x=X, y=Y, hue=Z, ci=None, fit_reg=False, scatter_kws={"s": 10, "alpha":0.4}) # scatter plot with linear regression for each category Z 
    # splot2 = sns.lineplot(data=df_plot, x=X, y=Y, hue=Z, marker="o")

    # g.map_dataframe(sns.scatterplot, x=x, y=y, hue=z)


    # # sfig1 = splot1.get_figure()
    # # sfig2 = splot2.get_figure()

    # SET PLOT TITLE
    # plt.title("Density vs Temperature ALD Thin Films")

    # SET AXIS LABELS
    # plt.xlabel("Deposition Temperature (°C)")
    # plt.ylabel("Density $(g.cm^{-3})$")

    sns.move_legend(fig, "lower center", bbox_to_anchor=(0.5, 1), ncol=2, frameon=False)
    return fig

fig1 = get_fit_plot(df_plot, X, Y, Z)
fig2 = get_line_plot(df_plot2, X, Y, Z)


# plt.show()

# bbox_inches stops title from being cut off in the plot
fig1.savefig('plots/plot1.png', dpi=150, bbox_inches="tight")
fig2.savefig('plots/plot2.png', dpi=150, bbox_inches="tight")
# splot2.savefig('plots/plot4.png', dpi=150, bbox_inches="tight")
