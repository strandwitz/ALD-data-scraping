import pandas as pd
import numpy as np
import re


def create_latex_labels(label):
    if type(label) is float and np.isnan(label):
        return label

    label = str(label)
    lbls = label.split(" ", 1)
    new_label = []
    for lbl in lbls:
        lbl = str(lbl)
        nlbl =  re.sub(r'([a-zA-Z])(\d+)', r'\1_{\2}',lbl)
        nlbl = lbl if (lbl==nlbl) else f"${nlbl}$"
        new_label.append(nlbl)

    label = " ".join(new_label)
    label = label.strip()
    label = rf'{label}'
    return label



def data_value_counts_num(df, z, n=1, how="=="):
    vcounts = df[z].value_counts()
    filt_vcounts = vcounts[vcounts == n]

    if how == "==": filt_vcounts = vcounts[vcounts == n]
    elif how == "<": filt_vcounts = vcounts[vcounts < n]
    elif how == "<=": filt_vcounts = vcounts[vcounts <= n]
    elif how == ">": filt_vcounts = vcounts[vcounts > n]
    elif how == ">=": filt_vcounts = vcounts[vcounts >= n]
    else: print(f"{how} not valid. how = [==,<,<=,>,>=]")

    filt = df[z].isin(filt_vcounts.index)
    df_vc = df.loc[filt, :][z].value_counts()

    return filt, df_vc