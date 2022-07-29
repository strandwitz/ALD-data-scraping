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

# TODO: option to choose material. Al2O3, TiO2


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


    fig = sns.lmplot(data=df, x=x, y=y, 
        hue=hue, hue_order=[True, False], markers=["^", "v"], palette=["darkorange", "dodgerblue"],
        ci=None, fit_reg=False,
        scatter_kws={"s": 45, "alpha":0.7}, 
        line_kws={"lw":1.5, "alpha":0.5})

    sns.regplot(data=df, x=x, y=y, scatter=False, ci=None, 
        line_kws={"lw":1.5, "alpha":0.5, "color":"k"}).set(title=daz.create_latex_labels(kwargs.get("title")))

    return fig


fig = plot_data(df_material, x=x_tdep, y=y_density, hue="PEALD?", title=INPUT_MATERIAL)
fig.savefig('plots/plot4.png', dpi=300, bbox_inches="tight")
