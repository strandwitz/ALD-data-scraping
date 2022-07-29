# Daria Zachariassen
# July 28, 2022

"""
ALD Data Scraping Project
"""

import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FuncFormatter
import random
import re

import daz_utils as daz # useful functions


sheet_id = "1CnYIYPMymwAKaVlElBk4ceNN2RtTxpKIHTzuTRjfY3s"
sheet_name = "FilmProps"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"


# LOAD DATA
df = pd.read_csv(url)

# CLEAN DATA
df.dropna(axis='columns', how='all', inplace=True) # drop columns that are all NA values

df.columns = df.columns.str.replace(r'[ ]{2,}',' ', regex=True) # make sure column names only have 1 space between words
df.columns = df.columns.str.strip() # Remove leading and trailing whitespace


# VIEW DATA
print(list(df.columns), "\n") # view column names
print(df.info(), "\n") # view column names and types
print(df.head(), "\n") # view the first few rows of the dataframe


# RENAME COLUMNS
cols = {"Tdep (°C)": "Deposition Temperature (°C)", 
        "Density (g.cm-3)": "Density $(g.cm^{-3})$"}

# VARIABLES
x_tdep="Deposition Temperature (°C)"
y_density="Density $(g.cm^{-3})$"
z_material="Material"

INPUT_MATERIAL = "Al2O3"

df.rename(columns = cols, inplace=True)

# print(df[z_material].value_counts(), "\n")

df[z_material] = df[z_material].str.strip() # Remove leading and trailing whitespace

# FILTER
# print("---------------------------------------------------")

filt_material = df[z_material].str.contains(INPUT_MATERIAL)
df_material = df.loc[filt_material]
print(df_material.info())


# PLOT

def plot_data(df, x, y, hue, **kwargs):
    cl="darkslateblue"

    fig = sns.lmplot(data=df, x=x, y=y, 
        hue=hue, hue_order=[True, False], markers=["^", "v"], palette=["orangered", "navy"],
        ci=None, fit_reg=False,
        scatter_kws={"s": 45, "alpha":0.5}, 
        line_kws={"lw":1.5, "alpha":0.5})

    sns.regplot(data=df, x=x, y=y, scatter=False, ci=None, 
        line_kws={"lw":1.5, "alpha":0.7, "color":cl}).set(title=daz.create_latex_labels(kwargs.get("title")))

    return fig

def plot_data2(df, x, y, hue, **kwargs):
    c1="darkslateblue"
    c2="darkslateblue"

    g = sns.FacetGrid(
        data=df,
        hue=hue, hue_order=[True, False],
        height=4, aspect=1.25,
        hue_kws={"edgecolor": [c1,c2], 'facecolor':["none", c2], "markers":["^", "v"]},
    )
    g.map(sns.scatterplot, x, y, linewidth=1.5, alpha=1)
    g.add_legend()


    sns.regplot(data=df, x=x, y=y, scatter=False, ci=None, 
        line_kws={"lw":1.5, "alpha":0.9, "color":c2}).set(title=daz.create_latex_labels(kwargs.get("title")))


    return g

fig = plot_data(df_material, x=x_tdep, y=y_density, hue="PEALD?", title=INPUT_MATERIAL)
fig.savefig('plots/plot4.png', dpi=300, bbox_inches="tight")


fig2 = plot_data2(df_material, x=x_tdep, y=y_density, hue="PEALD?", title=INPUT_MATERIAL)
fig2.savefig('plots/plot5.png', dpi=300, bbox_inches="tight")
