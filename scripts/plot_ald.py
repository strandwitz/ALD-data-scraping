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
import re

import daz_utils as daz



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

# TODO: in progess
z_material = Z
x_tdep = X

_, s_one = daz.data_value_counts_num(df, z_material, n=2, how="<")
_, s_two = daz.data_value_counts_num(df, z_material, n=2, how=">=")

print()
print(f"# of materials with 1 data point: {s_one.size}")
print(f"# of materials with 2+ data points: {s_two.size}")


minpoints_z = 2 # only plot if there are more than m values per category z_material
minpoints_x = 2 # only plot if there are more than m values per category z_material

# require a material to have more than minpoints_z data points to plot it
count_z = df[z_material].value_counts()
filt_minpoints_z = df[z_material].isin(count_z[count_z > minpoints_z].index)

# require more than minpoints_x unique temperatures per material to plot
count_x = df.groupby(z_material)[x_tdep].nunique()
filt_minpoints_x = df[z_material].isin(count_x[count_x > minpoints_x].index)

filt_minpoints = filt_minpoints_z & filt_minpoints_x

# each material has at least minpoints_z entries and minpoints_x different x values
df_plot = df.loc[filt_minpoints, :] 

# each material has at least minpoints_z entries
df_plot2 = df.loc[filt_minpoints_z, :]

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
    sns.reset_orig()

    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


    fig, ax = plt.subplots(figsize=(8.5,11)) # figsize=(6, 7)
    ax.grid(True, color = '#e8e8e6', linestyle = '--', linewidth = 0.5)
    # ax = fig.add_subplot()
    # ax = sns.lineplot(data=df, x=x, y=y, hue=z, marker="o", ci=None, markersize=4, alpha = 0.9, linestyle='')
    # sns.lineplot(data=df, x=x, y=y, ax=ax, sort=False, hue=z, dashes=False, alpha=0.3, legend=False, ci=None)
    order_z = list(df.sort_values(by=[Z], ascending=True)[Z].unique())
    # random.Random(2).shuffle(order_z)
    print(order_z)

    g2 = sns.lineplot(ax=ax, data=df, x=x, y=y, hue=z, alpha=0.4, lw=1, ci=None, legend=False, hue_order=order_z)
    g1 = sns.scatterplot(ax=ax, data=df, x=x, y=y, hue=z, alpha=0.7, s=35, style=z, hue_order=order_z, style_order=order_z)

    # SET PLOT TITLE
    plt.title("Density vs Temperature ALD Thin Films")

    # SET AXIS LABELS
    plt.xlabel("Deposition Temperature (°C)")
    plt.ylabel("Density $(g.cm^{-3})$")


    # df.groupby([x,z]).count()[y].unstack().plot(ax=ax, marker="o", markersize=4, alpha=0.6)

    # plt.scatter(df[x], df[y])

    # for label, df_g in df.groupby(z):
    #     df_g.vals.plot(kind="kde", ax=ax, label=label)

    handles, labels  =  ax.get_legend_handles_labels() # get legend text  
    labels = [daz.create_latex_labels(l) for l in labels] # Al2O3 --> $Al_{2}O_{3}$
    ax.legend(handles, labels) # set modified labels

    sns.move_legend(ax, title=f"{z}s", loc=2, bbox_to_anchor=(1.01, 1));


    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))

    return fig

def get_fit_plot(df, x, y, z, hue):
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
    
    
    # sns.set(font_scale = 1.50)
    df_plot = df.loc[:,[x,y,z,hue]]

    df_plot[z] = df_plot[z].apply(daz.create_latex_labels)

    SMALL_SIZE = 16
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 20

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


    # scatter plot with linear regression for each category Z 
    fig = sns.lmplot(data=df_plot, x=x, y=y, col=z, col_wrap=4, ci=None, 
        hue=hue, hue_order=[True, False], markers=["^", "v"], palette=["orangered", "navy"],
        facet_kws = dict(sharex=False, sharey=False), 
        scatter_kws={"s": 70, "alpha":0.5}, 
        line_kws={"lw":1.5, "alpha":0.5})

    
    # splot2 = sns.lmplot(data=df_plot, x=X, y=Y, hue=Z, ci=None, fit_reg=False, scatter_kws={"s": 10, "alpha":0.4}) # scatter plot with linear regression for each category Z 
    # splot2 = sns.lineplot(data=df_plot, x=X, y=Y, hue=Z, marker="o")

    # g.map_dataframe(sns.scatterplot, x=x, y=y, hue=z)


    # # sfig1 = splot1.get_figure()
    # # sfig2 = splot2.get_figure()

    # SET PLOT TITLE
    #  set title of each subplot to col value
    fig.set_titles(col_template="{col_name}")

    # fix y axis sigfigs
    for ax in fig.axes.flat:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, p: f'{y:.1f}'))

    # SET AXIS LABELS
    fig.set_axis_labels( "Deposition Temperature (°C)", "Density $(g.cm^{-3})$" )

    # CUSTOMIZE LEGEND
    sns.move_legend(fig, "lower center", bbox_to_anchor=(0.5, 1), ncol=2, frameon=False)

    # customize plot spacing and padding
    plt.subplots_adjust(wspace = 0.19, hspace=0.2)
    # plt.tight_layout()
    return fig

fig1 = get_fit_plot(df_plot, X, Y, Z, hue="PEALD?")
fig2 = get_line_plot(df_plot2, X, Y, Z)


# plt.show()

# bbox_inches stops title from being cut off in the plot
fig1.savefig('plots/plot1.png', dpi=200, bbox_inches="tight")
fig2.savefig('plots/plot2.png', dpi=300, bbox_inches="tight")
# splot2.savefig('plots/plot4.png', dpi=150, bbox_inches="tight")
