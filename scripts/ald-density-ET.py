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

# LOAD DATA
df_exp = pd.read_csv(url_exp)
df_thr = pd.read_csv(url_thr)

# CLEAN DATA
df_exp.dropna(axis='columns', how='all', inplace=True) # drop columns that are all NA values
df_exp.columns = df_exp.columns.str.replace(r'[ ]{2,}',' ', regex=True) # make sure column names only have 1 space between words

df_thr.dropna(axis='columns', how='all', inplace=True) # drop columns that are all NA values
df_thr.columns = df_thr.columns.str.replace(r'[ ]{2,}',' ', regex=True) # make sure column names only have 1 space between words


# VIEW DATA
print(list(df_exp.columns), "\n") # view column names
print(df_exp.info(), "\n") # view column names and types
print(df_exp.head(), "\n") # view the first few rows of the dataframe

print("-"*30)

print(list(df_thr.columns), "\n") # view column names
print(df_thr.info(), "\n") # view column names and types
print(df_thr.head(), "\n") # view the first few rows of the dataframe
