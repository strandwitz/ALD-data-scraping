"""
Plot Experimental vs Theoretical Densities

https://docs.google.com/spreadsheets/d/1CnYIYPMymwAKaVlElBk4ceNN2RtTxpKIHTzuTRjfY3s/edit#gid=1436201923
"""


import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import random
# from adjustText import adjust_text # automatic text placement in plots
import re

sheet_name_exp = "FilmProps"
sheet_name_thr = "TheorDensity"
sheet_id = "1CnYIYPMymwAKaVlElBk4ceNN2RtTxpKIHTzuTRjfY3s"
url_exp = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name_exp}"
url_thr = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name_thr}"

# url_exp = "data/data_exp.csv"
# url_thr = "data/data_thr.csv"

def ald_print_info(df, label="", info=True, head=[True, 5], tail=[False, 5], cols=True, vc=[False, None]):
    # VIEW DATA
    print()
    print(">"*60, f" {label}") # visual separator on terminal
    if cols:
        print(list(df.columns), "\n") # view column names

    if info:
        print(df.info(), "\n") # view column names and types

    if head[0]:
        print(df.head(head[1]), "\n") # view the first few rows of the dataframe

    if tail[0]:
        print(df.tail(tail[1]), "\n") # view the last few rows of the dataframe

    if vc[0]:
        print(df[vc[1]].value_counts())

    print("<"*60) # visual separator on terminal


def ald_load_data(urls, view_data=True):

    df = 0
    for url in urls:
        df = pd.read_csv(url)

        # CLEAN DATA
        df.dropna(axis='columns', how='all', inplace=True) # drop columns that are all NA values
        df.columns = df.columns.str.replace(r'[ ]{2,}',' ', regex=True) # make sure column names only have 1 space between words

        if view_data:

            # VIEW DATA
            sheet = url.split("sheet=", 1)[1]
            ald_print_info(df, label=sheet)

        yield df


data_generator = ald_load_data([url_exp, url_thr], view_data=False)



# PRINT COLUMN NAMES
df_exp = next(data_generator)
exp_all_columns = list(df_exp.columns)
print("EXPERIMENTAL ",exp_all_columns)


df_thr = next(data_generator)
thr_all_columns = list(df_thr.columns)
print("THEORETICAL ",thr_all_columns)


# RENAME DENSITY COLUMNS
density_exp = "Experimental Density"
density_thr = "Theoretical Density"
density_units = "$(g.cm^{-3})$"

re_cols = {
    "exp": {
        "Density (g.cm-3)": density_exp,
        "Tdep (°C)": "Deposition Temperature (°C)"
    }, 
    "thr": {"Density (g.cm-3)": density_thr}
}

df_exp.rename(columns=re_cols['exp'], inplace=True)
df_thr.rename(columns=re_cols['thr'], inplace=True)


# CHOOSE KEY
key = "Material"

# SELECT COLUMNS
cols_exp = ['Source DOI', key, density_exp, 
            'Deposition Temperature (°C)', 'Tpostdep', 'PEALD?', 'Density Meas',
            'k', 'thickness of meas. k (nm)', 'RF Bias (V)', 
            # 'P1 (metal)', 'P2 (nonmetal)', 'P3 (optional)', 'P4',
            'Paper also measured']
df_exp = df_exp[cols_exp]

cols_thr = [key, 'Phase', density_thr, "Label", 'Space group']
df_thr = df_thr[cols_thr]

# TODO SAVE DATA TO CSV 
# TODO ADD TRY EXCEPT 


# GET DATA TO PLOT

def ald_merge_df(df1, df2, key, cols, view_data=True):
    exp_density = df1.loc[:, [key, cols[0][0]]]
    thr_density = df2.loc[:, [key, cols[1][0]]]
    thr_phase = df2.loc[:, [key, *cols[1]]]

    df_merged = pd.merge(exp_density, thr_density, on=key)
    df_merged = pd.merge(df_merged, thr_phase, on=[key, cols[1][0]])

    if view_data:
        ald_print_info(df_merged, label=" MERGED")

    return df_merged


# MERGE DATAFRAMES
cols = ([density_exp], [density_thr, "Phase", "Label"])
df_merged = ald_merge_df(df_exp, df_thr, key, cols, view_data=False)



# SORT VALUES
def sort_valuecounts(df, key):
    df = df.loc[:] # prevent SettingWithCopyWarning
    df['freq'] = df.groupby(key)[key].transform('count')
    df.sort_values(['freq',key], ascending=False, inplace=True)
    df.drop('freq', axis=1, inplace=True)

    df.reset_index(drop=True, inplace=True)

    return df


df_exp = sort_valuecounts(df_exp, key)
df_thr = sort_valuecounts(df_thr, key)
df_merged = sort_valuecounts(df_merged, key)


# PRINT DF INFO
ald_print_info(df_exp, label="DF EXPERIMENTAL", vc=[False, key])
ald_print_info(df_thr, label="DF THEORETICAL", vc=[False, key])
ald_print_info(df_merged, label="DF MERGED", vc=[False, key], head=[False, 0], tail=[True, 10])


def create_latex_labels(label, bold=False):
    if type(label) is float and np.isnan(label):
        return label

    label = str(label)
    lbls = label.split(" ", 1)
    new_label = []
    for lbl in lbls:
        lbl = str(lbl)
        nlbl =  re.sub(r'([a-zA-Z])(\d+)', r'\1_{\2}',lbl)
        if bold:
            nlbl = lbl if (lbl==nlbl) else rf"$\mathbf{{{nlbl}}}$"
        else:
            nlbl = lbl if (lbl==nlbl) else f"${nlbl}$"
        new_label.append(nlbl)

    label = " ".join(new_label)
    label = label.strip()
    label = rf'{label}'
    return label

# test fucntion
# print("*"*40,create_latex_labels("NiO"))
# print("*"*40,create_latex_labels("Al2O3"))
# print("*"*40,create_latex_labels(r"Al_{2}O_{3}"))


# PLOT

def plot_data(df, x, y, z):
    print("PLOT", "-"*60)

    fig, ax = plt.subplots(figsize=(16,9)) # figsize=(6, 7)
    # ax.grid(True, color = '#e8e8e6', linestyle = '--', linewidth = 0.5)
    # ax = fig.add_subplot()
    # ax = sns.lineplot(data=df, x=x, y=y, hue=z, marker="o", ci=None, markersize=4, alpha = 0.9, linestyle='')
    # sns.lineplot(data=df, x=x, y=y, ax=ax, sort=False, hue=z, dashes=False, alpha=0.3, legend=False, ci=None)
    order_z = list(df.sort_values(by=[z], ascending=True)[z].unique())
    # random.Random(2).shuffle(order_z)
    # print(order_z)

    # g2 = sns.lineplot(ax=ax, data=df, x=x, y=y, hue=z, alpha=0.4, lw=1, ci=None, legend=False, hue_order=order_z)
    ax.axline((0,0), slope=1, color='silver', lw=1, ls='--', alpha=0.5, label='_none_')
    g1 = sns.scatterplot(ax=ax, data=df, x=x, y=y, hue=z, alpha=0.6, s=35, style=z, hue_order=order_z, style_order=order_z)
    g2 = sns.scatterplot(ax=ax, data=df, x=x, y=x, color="black", alpha=0.9, s=15, marker="+")

    # filt = True
    cols = [z, x]
    # print(df.head())
    df2 = df.loc[:,[x,y,z]]
    df2.sort_values(by=[x,y], ascending=True, inplace=True)
    df2.drop_duplicates(subset=cols, keep='last', inplace=True)
    # print(df2.head())
    # print(df2.shape)
    for line in range(0,df2.shape[0]):
        ox=0
        oy=0.1
        px=df2[x].iloc[line]
        py=df2[y].iloc[line]
        ty=px
        tx=px

        # if line % 3 == 0:
        #     ox = 0.5
        #     oy = 0.75
        # elif line % 3 == 1:
        #     ox = 0.5
        #     oy = 1.25
        # elif line % 3 == 2:
        #     ox = 0.5
        #     oy = 1.75
        # else:
        #     ox = 1.0
        #     oy = 0

        # if (py < px) and (px-py < 0.5):
        #     ox = -0.5
        #     oy = oy*-1
        #     # py=px

        if py+oy > 12.2: 
            oy = 0.3
            ox = 0.5

        plt.annotate(df2[z].iloc[line], 
            xy=(tx, ty), 
            xytext=(px-ox, py+oy), 
            horizontalalignment='center', verticalalignment='bottom', size=8, color='black', weight='light',
            arrowprops={"arrowstyle":"->, widthA=.5, widthB=.5", "color":"gray", "alpha":0.4})


    #place legend outside top right corner of plot
    # plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))


    return fig

def plot_swarm(df, x, y, z):
    fig = plt.figure(figsize=(16,9))

    # fig, ax = plt.subplots(figsize=(10,8)) # figsize=(6, 7)

    order_z = list(df.sort_values(by=[z], ascending=True)[z].unique())
    g1 = sns.scatterplot(data=df, x=y, y=y, ci=None, hue=x, alpha=0.6, s=35, style=z, style_order=order_z)



    # g1 = sns.catplot(data=df, x=y, y=y, hue=x, col=z, col_wrap=7, alpha=0.8, sharex=False, sharey=False, kind="strip", jitter=False, height=5, aspect=1.25, ci=None)
    # # plt.xticks(rotation=90)

    # # ax1 = g1.axes[0]

    # # g1.set(ylim=(0,None), xlim=(0,None))

    # for ax in g1.axes.flatten():
    #     ax.axline((0,0), slope=1, color='silver', lw=1, ls='--', label='_none_')
    #     ax.tick_params(labelbottom=True, labelleft=True)

    #     ax.set(xlim=(0,12))  # x axis starts at 0
    #     ax.set(ylim=(0,12))  # y axis starts at 0




    return fig

def plot_data1(df, x, y, z, point_labels,  **kwargs):
    ald_print_info(df, label="DF PLOTTING")

    fig, ax = plt.subplots(figsize=(13,10)) # figsize=(6, 7)

    order_z = list(df[z].unique())

    # formated_labels = df[z].apply(create_latex_labels)
    # print(formated_labels)
    # print(df[z].value_counts())

    ax.axline((0,0), slope=1, color='silver', lw=1, ls='--', alpha=0.5, label='_none_')
    g1 = sns.scatterplot(ax=ax, data=df, x=x, y=y, hue=z, alpha=0.6, s=35, style=z, hue_order=order_z, style_order=order_z)
    g2 = sns.scatterplot(ax=ax, data=df, x=x, y=x, color="black", alpha=0.9, s=15, marker="+")

    plt.minorticks_on()

    # SET TITLE
    plt.title("ALD Thin Film Density")

    # SET AXIS LABELS
    ax.set_xlabel(" ".join([x, kwargs.get('units')]))
    ax.set_ylabel(" ".join([y, kwargs.get('units')]))


    # PLOT LABELS
    bold_labels = True
    df_lbls = df.loc[:, [x, y, z, point_labels]]
    df_lbls.sort_values(by=[x,y], ascending=[True,True], inplace=True)
    df_lbls.drop_duplicates(subset=[x, z], keep='last', inplace=True)

    # df_lbls["tmp"] =  df_lbls[point_labels[1]].str.contains( df_lbls[point_labels[0]], regex=False )
    df_lbls[point_labels] = df_lbls[point_labels].fillna('')
    # filt_material_in_phase = df_lbls.apply(lambda r: r[point_labels[0]] in r[point_labels[1]], axis=1)

    # for lbl_col in point_labels:
        # print(df_lbls[lbl_col].head())
        # df_lbls[lbl_col] = df_lbls[lbl_col].apply(create_latex_labels)
    
    df_lbls[point_labels] = df_lbls[point_labels].apply(create_latex_labels, bold=bold_labels)
    # df_lbls[point_labels[0]] = df_lbls[point_labels[0]].apply(create_latex_labels)

    # df_lbls.loc[filt_material_in_phase, ["label"]] = df_lbls[point_labels[1]] # NOTE: may need to split str if phase has multiple annotations
    # df_lbls.loc[~filt_material_in_phase, ["label"]] = df_lbls[point_labels[0]]+"\n"+df_lbls[point_labels[1]].str.replace(r'\s', r'\n', n=1, regex=True)
    df_lbls[point_labels] = df_lbls[point_labels].str.replace(r'\s+', r'\n', n=2, regex=True)
    df_lbls[point_labels] = df_lbls[point_labels].str.strip()
    df_lbls["label_lines"] = df_lbls[point_labels].str.count("\n")+1

    filt_twolines = df_lbls["label_lines"] == 2
    long_label = 6
    df_lbls.loc[filt_twolines, ["len_twolines"]] = df_lbls[point_labels].str.replace(r'.+\n(.+)(?:\n.*)?', r'\1', regex=True).str.len() > long_label
    df_lbls["len_twolines"] = df_lbls["len_twolines"].fillna(False)

    df_lbls["below_line"] = df_lbls[x] > df_lbls[y]
    df_lbls["seq_above_below"] = df_lbls.groupby("below_line").cumcount()+1
    df_lbls["seq_lbl_lines"] = df_lbls.groupby(["below_line", "label_lines", "len_twolines"]).cumcount()+1

    # df_lbls["label"] = df_lbls[point_labels[1]]
    # else:
    #     df_lbls["label"] = df_lbls[point_labels[0]]+"\n"+ df_lbls[point_labels[1]].str.replace(r'\s', r'\n', n=1, regex=True)

    print(df_lbls.head(20))
    print(df_lbls.shape)

    texts = []

    for line in range(0,df_lbls.shape[0]):
        # GET X AND Y COORDS 
        x_thr=df_lbls[x].iloc[line]
        y_thr=df_lbls[x].iloc[line] # theoretical x and y are the same
        y_exp=df_lbls[y].iloc[line] # experimental y value

        below_line = df_lbls["below_line"].iloc[line] # boolean

        seq_above_below = df_lbls["seq_above_below"].iloc[line] # sequence above and below line
        seq_lbl_lines = df_lbls["seq_lbl_lines"].iloc[line] # sequence number of lines


        # GET LABEL TEXT
        pnt_lbl = df_lbls[point_labels].iloc[line]

        nlines = df_lbls["label_lines"].iloc[line]
        oneline = nlines == 1
        twolines = nlines == 2
        manylines = nlines >= 3
        # print(nlines)

        lg_twolines = df_lbls["len_twolines"].iloc[line]
        sm_twolines = twolines and not(lg_twolines)


        # CREATE SECTIONS FOR LABEL PLACEMENT
        # PLOT MARGINS
        mxh = 0.10 # 13% margins
        mxl = 0.04 # 2% margins
        myh = 0.04 # 6% margins
        myl = 0.04 # 2% margins

        delta_x = ax.get_xlim()[1] - ax.get_xlim()[0]
        delta_y = ax.get_ylim()[1] - ax.get_ylim()[0]

        margin_x_high = delta_x * mxh
        margin_x_low = delta_x * mxl
        margin_y_high = delta_y * myh
        margin_y_low = delta_y * myl

        lim_y_low = ax.get_ylim()[0] + (margin_y_low)
        lim_y_high = ax.get_ylim()[1] - (margin_y_high)

        lim_x_low = ax.get_xlim()[0] + (margin_x_low)
        lim_x_high = ax.get_xlim()[1] - (margin_x_high)

        # create margin booleans
        point_in_margin_x_low = x_thr <= lim_x_low
        point_in_margin_x_high = x_thr >= lim_x_high

        point_in_margin_y_low = y_exp <= lim_y_low
        point_in_margin_y_high = y_exp >= lim_y_high


        # PLOT SECTION X AXIS
        num_sections = 6.0
        sec_x1 = delta_x / num_sections
        sec_x2 = ax.get_xlim()[1] - sec_x1

        sections = [sec_x1 * i for i in range(0,int(num_sections)+1)]
        point_in_section_more = [x_thr > x for x in sections]
        point_section = point_in_section_more.count(True)
        # print(sections)
        # print(x_thr)
        # print(point_in_section_more)
        # print(point_section)


        point_in_section_first = point_section == 1
        point_in_section_last = point_section == num_sections
        point_in_section_middle = not(point_in_section_first or point_in_section_last)


        y_txt = y_exp
        x_txt = x_thr
        rad = 0
        halign = "center"
        valign = "center"
        text_weight = "bold" if bold_labels else "normal"
        text_color = "black"
        arrow_color = "gray"

        # # ADJUST TEXT PLACEMENT OFFSETS


        if below_line:
            x_txt = x_thr
            y_txt = 0
            # mmm=1.7

            slider = 0.0
            y_scatter=0
            equ_x_offset=8.5 # x offset
            pnt_scalor=1.0 # expansion scale
            y_line=0.5 # line to expand away from
            equ_slope=1.15 # slope


            halign = "center"
            valign = "bottom"
            flip_arrow = False

            # even = ~((seq_lbl_lines%2) or -1)
            if oneline:
                # print(seq_lbl_lines, y_exp, pnt_lbl)
                equ_x_offset=7.0 # x offset
                scatter_shift=8
                y_scatter = ((seq_lbl_lines+scatter_shift)%9*0.38)
                # flip_arrow=True
                # pnt_scalor=2.0 # expansion scale
                # y_line=1.0 # line to expand away from
                equ_slope=2.0 # slope

                # halign = "left"
                valign = "bottom"

            elif sm_twolines:
                # print(seq_lbl_lines, y_exp, pnt_lbl)
                equ_x_offset=11.0 # x offset
                scatter_shift=1
                y_scatter = ((seq_lbl_lines+scatter_shift)%3*1.0)
                equ_slope=2.0 # slope
                # pnt_scalor=2.5 # expansion scale
                # y_line=0.5 # line to expand away from
                # equ_slope=mmm # slope

                # rad = -0.05
                # halign = "left"
                valign = "bottom"

            elif lg_twolines:
                equ_x_offset=14.5 # x offset
                scatter_shift=2
                y_scatter = ((seq_lbl_lines+scatter_shift)%4*0.95)
                equ_slope=2.25 # slope

                # pnt_scalor=2.5 # expansion scale
                # y_line=0.5 # line to expand away from

                # rad = -0.05
                # halign = "left"
                valign = "bottom"


            y_base_values = x_thr
            b=y_line * (1-pnt_scalor)
            u=(pnt_scalor *(y_base_values)) + b

            yy=u/equ_slope + equ_x_offset + y_scatter
            xx=u

            if flip_arrow:
                yy=u/equ_slope + (-1*equ_x_offset) + y_scatter
                # xx, yy = yy, xx
                # yy = -yy

            x_txt = xx + slider
            y_txt = yy + slider

        else: # above line
            x_txt = x_thr
            y_txt = 0

            halign = "center"
            valign = "bottom"
            # text_weight = "bold"
            text_color = "red"
            arrow_color = "red"

            slider=0
            y_scatter=0
            equ_y_offset=3.5 # x offset
            pnt_scalor=1.0 # expansion scale
            x_line=0.5 # line to expand away from
            equ_slope=1.1 # slope

            # even = ~((seq_lbl_lines%2) or -1)
            if oneline:
                # print(seq_lbl_lines, y_exp, pnt_lbl)
                equ_y_offset=1.4 # x offset
                y_scatter = (seq_lbl_lines%2 * 0.75)
                # slider=1.0
                # pnt_scalor=1.0 # expansion scale
                # x_line=1.0 # line to expand away from
                equ_slope=0.95 # slope

                # halign = "right"
                valign = "bottom"

            elif sm_twolines:
                print(seq_lbl_lines, y_exp, pnt_lbl) # ON
                equ_y_offset=3.5 # x offset
                scatter_shift = 2
                y_scatter = ((seq_lbl_lines+scatter_shift)%3 * 0.7)
                # slider=5.0
                # pnt_scalor=4.0 # expansion scale
                # x_line=2.0 # line to expand away from
                equ_slope=1.2 # slope

                # rad = -0.05
                # halign = "center"
                valign = "bottom"

            elif lg_twolines:
                equ_y_offset=8.7 # x offset
                scatter_shift = 4
                y_scatter = ((seq_lbl_lines+scatter_shift)%5*1.0)
                # pnt_scalor=2.5 # expansion scale
                # x_line=0.5 # line to expand away from
                equ_slope=2.8 # slope

                # rad = -0.05
                # halign = "left"
                valign = "bottom"

            x_base_values = x_thr
            b=x_line * (1-pnt_scalor)
            u=(pnt_scalor *(x_base_values)) + b

            xx=u
            yy=u/equ_slope + equ_y_offset + y_scatter

            x_txt = xx + slider
            y_txt = yy + slider

        # BRING ANNOTATIONS INSIDE THE MARGINS OF THE PLOT
        while y_txt > lim_y_high:
            # print(f"\nH {y_exp+oy:.4f} > lim_y_high {lim_y_high:.4f} -- {oy}")
            if abs(round(y_txt, 5)) <= abs(round(lim_y_high, 5)):
                y_txt = lim_y_high

            if lim_x_high < 0:
                y_txt *= 1.01
            else:
                y_txt *= 0.99

        while x_txt > lim_x_high:
            # print(f"\nH {x_txt:.4f} > lim_x_high {lim_x_high:.4f} -- {ox}")
            if abs(round(x_txt, 5)) <= abs(round(lim_x_high, 5)):
                x_txt = lim_x_high

            if lim_x_high < 0:
                x_txt *= 1.01
            else:
                x_txt *= 0.99

        while y_txt < lim_y_low:
            # print(f"\nL {y_txt:.4f} < lim_y_low {lim_y_low:.4f}")
            if abs(round(y_txt, 5)) >= abs(round(lim_y_low, 5)):
                y_txt = lim_y_low

            if lim_y_low < 0:
                y_txt *= 0.995
            else:
                y_txt *= 1.005

        while x_txt < lim_x_low:
            # print(f"\nL {x_thr-ox:.4f} < lim_x_low {lim_x_low:.4f}")
            if abs(round(x_txt, 5)) >= abs(round(lim_x_low, 5)):
                x_txt = lim_x_low

            if lim_x_low < 0:
                x_txt *= 0.995
            else:
                x_txt *= 1.005


        pnt_coords = (x_thr, y_exp)
        text_coords = (x_txt, y_txt)

        texts.append(plt.annotate(pnt_lbl, 
                    xy=pnt_coords, xytext=text_coords, 
                    verticalalignment=valign, horizontalalignment=halign, 
                    size=7, color=text_color, weight=text_weight,
                    arrowprops={"arrowstyle":"-|>, widthA=.4, widthB=.4",
                                "connectionstyle":f"arc3,rad={rad}",
                                "color":arrow_color, "alpha":0.2
                    })
        )


    # adjust_text(texts, precision=0.001,
    #     force_text=(0.01, 0.1), 
    #     force_points=(0.01, 0.1),
    #     expand_text=(1.01, 1.01), 
    #     expand_points=(1.01, 1.01),
    #     autoalign='y', avoid_points=False, avoid_self=False,
    #     only_move={'points':'', 'text':'xy', 'objects':''})
    


    # LEGEND
    handles, labels  =  ax.get_legend_handles_labels() # get legend text  
    labels = [create_latex_labels(l) for l in labels] # Al2O3 --> $Al_{2}O_{3}$
    ax.legend(handles, labels) # set modified labels

    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), ncol=2)

    return fig



fig2 = plot_data1(df_merged, x=density_thr, y=density_exp, z=key, point_labels="Label", units=density_units)

fig2.savefig('plots/plotDensities-4.png', dpi=300, bbox_inches="tight")

# plt.show()

